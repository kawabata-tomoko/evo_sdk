
import torch
import torch.nn as nn

from evo.srcs.BaseModel.StripedHyena.component.ParallelGatedMLP import ParallelGatedMLP
from evo.srcs.BaseModel.StripedHyena.component.RMSNorm import RMSNorm
from evo.srcs.BaseModel.StripedHyena.component.ParallelHyenaFilter import ParallelHyenaFilter


class ParallelGatedConvBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.low_mem_mode = config.get("low_mem_mode", False)
        dtype = config.get("hyena_block_dtype", torch.float32)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.pre_norm, self.post_norm = RMSNorm(config).to(dtype=dtype), RMSNorm(config).to(dtype=dtype)
        self.filter = ParallelHyenaFilter(config, layer_idx).to(dtype=dtype)
        self.projections = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_filter_dense = nn.Linear(config.hidden_size, config.hidden_size).to(dtype)
        self.mlp = ParallelGatedMLP(config).to(dtype=mlp_dtype)

        self.proj_norm_fn = self.proj_norm
        self.res_mlp_norm_fn = self.res_mlp_norm

        if self.config.get("compile", False):
            self.proj_norm_fn = torch.compile(self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead")
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )

    def proj_norm(self, x):
        return self.projections(self.pre_norm(x))

    def res_mlp_norm(self, x):
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm_fn(u)

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z = z * padding_mask[..., None]

        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)

        z_in = self.out_filter_dense(z) + u

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm_fn(z_in)

        return y, inference_params
