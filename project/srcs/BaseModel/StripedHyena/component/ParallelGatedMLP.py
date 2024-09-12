import torch.nn as nn
import torch.nn.functional as F

def grab_first_if_tuple(x):
    return x[0] if x.__class__.__name__ == "tuple" else x

class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "silu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        if config.get("inner_mlp_size", None) is not None:
            inner_size = config.inner_mlp_size

        self.l1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)

