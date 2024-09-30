# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli
# Note: MP and PP utilities are removed for ease of use and editing.
import torch
import torch.nn as nn

from evo.srcs.BaseModel.inference.cache import (
    InferenceParams,
    RecurrentInferenceParams,
)
from evo.srcs.BaseModel.StripedHyena.component import (
    AttentionBlock,
    ParallelGatedConvBlock,
    ParallelHyenaFilter,
    RMSNorm,
    VocabParallelEmbedding,
    Embedding,
)
from evo.utils.utils import print_rank_0


def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hyena_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
    else:
        raise NotImplementedError


class StripedHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = VocabParallelEmbedding(config)
        # self.embedding_layer = Embedding(config)
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None
        self.unembed = (
            self.embedding_layer
            if config.tie_embeddings
            else VocabParallelEmbedding(config)
        )
        # self.unembed = self.embedding_layer if config.tie_embeddings else Embedding(config)

        if config.get("use_flashfft", "False"):
            try:
                from flashfftconv import FlashFFTConv
            except:
                raise ImportError
            self.flash_fft = FlashFFTConv(2 * config.seqlen, dtype=torch.bfloat16)
        else:
            self.flash_fft = None

        self.blocks = nn.ModuleList(
            get_block(config, layer_idx, flash_fft=self.flash_fft)
            for layer_idx in range(config.num_layers)
        )

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        L = x.shape[1]
        x = self.embedding_layer.embed(x)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(
                x, padding_mask=padding_mask
            )

        x = self.norm(x)
        if self.config.unembed==True:
            x = self.unembed.unembed(x)
        return x, inference_params_dict_out

    def stateful_forward(self, x, inference_params_dict=None):
        for block_idx, block in enumerate(self.blocks):
            x, _ = block(x, inference_params=inference_params_dict["mha" if block_idx in self.config.attn_layer_idxs else "hyena"])

        return x, inference_params_dict

    def stateless_forward(self, x, padding_mask=None):
        if type(padding_mask) == torch.Tensor:
            x = x * padding_mask[..., None]

        for _, block in enumerate(self.blocks):
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)
        return x, None

    def initialize_inference_params(self):
        print_rank_0("Initializing inference params...")
        return {
            "mha": InferenceParams(
                max_seqlen=self.config.get("max_seqlen", 8192),
                max_batch_size=self.config.get("max_batch_size", 1),
                seqlen_offset=0,
            ),
            "hyena": RecurrentInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                state_dim=self.config.state_size,
                seqlen_offset=0,
            ),
        }

    def precompute_filters(self, L, device):
        for block in self.blocks:
            if (
                type(block) == ParallelGatedConvBlock
                and type(block.filter) == ParallelHyenaFilter
            ):
                L = block.filter.long_fir_threshold or L
                print_rank_0(f"Precomputing filters, L={L}...")

                filter_dtype = torch.float16 if L >= 2048 else torch.float32

                block.filter._set_time(L, device)
                residues, poles = (
                    torch.view_as_complex(block.filter.residues.to(torch.float16)),
                    torch.view_as_complex(block.filter.poles.to(torch.float16)),
                )

                block.filter.h = (residues * poles**block.filter.t).real.sum(1)[None]
                block.filter.h = block.filter.h.to(dtype=filter_dtype)

    def load_poles_residues(self, path):
        "Load different poles and residues for each layer."
        for block_idx, block in enumerate(self.blocks):
            if (
                type(block) == ParallelGatedConvBlock
                and type(block.filter) == ParallelHyenaFilter
            ):
                print(f"Loading poles and residues for block {block_idx}")
                poles = torch.load(
                    f"{path}/approx_poles_{block_idx + 1}.pt", map_location="cpu"
                )
                poles = torch.view_as_real(poles)
                residues = torch.load(
                    f"{path}/approx_residues_{block_idx + 1}.pt",
                    map_location="cpu",
                )
                residues = torch.view_as_real(residues)
                poles = poles.permute(1, 0, 2).unsqueeze(-2)
                residues = residues.permute(1, 0, 2).unsqueeze(-2)

                block.filter.poles = nn.Parameter(poles)
                block.filter.residues = nn.Parameter(residues)
                residues = residues.permute(1, 0, 2).unsqueeze(-2)

                block.filter.poles = nn.Parameter(poles)
                block.filter.residues = nn.Parameter(residues)

    def to_bfloat16_except_poles_residues(self):
        """Convert all parameters to bfloat16 except for the poles and residues.

        Particularly important for longer prompts.
        """
        for k, p in self.named_parameters():
            if "poles" not in k and "residues" not in k:
                p.data = p.data.to(torch.bfloat16)

    def load_from_split_converted_state_dict(self, path):

        print("Loading from split converted state dict")

        embedding_weight = torch.load(f"{path}/layer_00.pt")["word_embeddings.weight"]
        self.embedding_layer.weight = nn.Parameter(
            embedding_weight.to(self.embedding_layer.weight.dtype)
        )

        print("Loading embedding weight ok")

        if self.config.get("final_norm", False) is not None:
            idx = len(self.blocks) + 1
            final_norm_scale = torch.load(f"{path}/layer_{idx:02d}.pt")["norm.scale"]
            self.norm.scale = nn.Parameter(final_norm_scale.to(self.norm.scale.dtype))

            print("loading final norm ok")

        if not self.config.get("tie_embeddings", True):
            idx = len(self.blocks) + 2
            embedding_weight = torch.load(f"{path}/layer_{idx:02d}.pt")[
                "word_embeddings.weight"
            ]
            self.unembed.weight = nn.Parameter(
                embedding_weight.to(self.unembed.weight.dtype)
            )

            print("loading unembed weight ok")

        # strict = False if type(block) == ParallelGatedConvBlock else True
        # some blocks (optionally) go through a round of conv distillation on some parameters
        strict = True  # safer to be strict and account for every layer

        for block_idx, block in enumerate(self.blocks):
            print(f"loading block {block_idx}...")
            loaded_dict = torch.load(f"{path}/layer_{block_idx + 1:02d}.pt")
            block.load_state_dict(loaded_dict, strict=strict)
