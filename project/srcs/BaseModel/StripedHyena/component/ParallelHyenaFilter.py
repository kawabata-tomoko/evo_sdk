
import torch
import torch.nn as nn
import torch.nn.functional as F
from project.srcs.BaseModel.inference.engine import HyenaInferenceEngine
from project.utils.utils import column_split


class ParallelHyenaFilter(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = config.get("hyena_filter_groups", self.config.hidden_size)

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.counter = 0
        self.column_split_hyena = config.get("column_split_hyena", True)

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        self.D = nn.Parameter(torch.zeros(self.hidden_size))

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        # after preprocessing here we can save the new checkpoint
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(torch.randn(3 * config.hidden_size, 1, config.short_filter_length))
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size)) if config.short_filter_bias else None
        )

        self.engine = HyenaInferenceEngine(layer_idx=layer_idx)
        self.use_flash_depthwise = config.get("use_flash_depthwise", False)
        self.data_dtype = None

        if self.use_flash_depthwise:
            self.fir_fn = FlashDepthwiseConv1d(#TODO: check this import.
                channels=3 * self.hidden_size,
                kernel_size=self.short_filter_length,
                padding=self.short_filter_length - 1,
                weights=self.short_filter_weight,
                bias=self.short_filter_bias,
                device=None,
                dtype=self.config.get("depthwise_dtype", torch.bfloat16),
            )
        else:
            self.fir_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False, "long_fir_threshold not compatible with fused flashfft"

        self.num_systems = self.hidden_size // self.hyena_filter_groups

        poles = torch.randn(self.num_systems, self.state_size, 1, 2)

        # TODO: bring over init from internals
        poles[..., 0] = 1e-2 * torch.randn(self.num_systems, self.state_size, 1)
        poles[..., 1] = 1e-3 * torch.randn(self.num_systems, self.state_size, 1)

        self.poles = nn.Parameter(poles)

        self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, 1, 2))
        self.h = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)

        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
        )
        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.h is None:
            h, filter_dtype, poles, residues = self.compute_filter(L, u.device)
        else:
            h = self.h
            filter_dtype = self.h.dtype

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 1)

        # if inference_params is not None, we plan to perform generation:
        # prefilling is handled by the engine.
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )
        y = self.engine.parallel_iir(
            z_pre,
            h,
            self.D,
            L,
            t=self.t,
            poles=self.poles,
            residues=self.residues,
            dims=dims,
            inference_params=inference_params,
            layer_idx=self.layer_idx,
            prefill_style=self.config.get("prefill_style", "fft"),
            use_flashfft=self.use_flashfft,
            fftconv_fn=self.fftconv_fn,
            column_split_hyena=self.column_split_hyena,
            long_fir_threshold=self.long_fir_threshold,
            padding_mask=padding_mask,
        )

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype
        if len(u.shape) > 2:
            u = u[:, -1]

        fir_state, iir_state = (
            inference_params.fir_state_dict[self.layer_idx],
            inference_params.state_dict[self.layer_idx],
        )

        z_pre, fir_state = self.engine.step_fir(
            u, fir_state, weight=self.short_filter_weight, bias=self.short_filter_bias
        )
        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        y, iir_state = self.engine.step_iir(
            x2,
            x1,
            v,
            self.D,
            self.residues,
            self.poles,
            iir_state,
            iir_groups=self.hyena_filter_groups,
        )

        inference_params.fir_state_dict[self.layer_idx] = fir_state
        inference_params.state_dict[self.layer_idx] = iir_state
        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        """
        Set [0, 1, ..., L-1] where L is the length of the current batch of inputs.
        If L is greater than the length of the previous batch, then the time vector is
        reinitialized. Otherwise, the time vector is truncated from cache.
        """
        if not hasattr(self, "t"):
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            torch.view_as_complex(self.residues.to(filter_dtype)),
            torch.view_as_complex(self.poles.to(filter_dtype)).log(),
        )
        h = (residues * (log_poles * self.t).exp()).real.sum(1)[None]
        return h, filter_dtype, log_poles, residues
