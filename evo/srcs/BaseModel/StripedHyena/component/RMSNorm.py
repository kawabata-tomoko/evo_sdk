import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            try:
                from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

                self.rmsnorm_func = rmsnorm_func
            except:
                raise ImportError(
                    "For `use_flash_rmsnorm`: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm`"
                )

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
        return self.scale * y
