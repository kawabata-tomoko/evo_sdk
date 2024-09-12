import torch
from torch import Tensor
import torch.nn as nn


class Embedding(nn.Module):
    _train_dtype = "bf16"

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

    def embed(self, input_ids, position_ids=None, tokentype_ids=None):
        return self.word_embeddings(input_ids)

    def unembed(self, u):
        weight = self.word_embeddings.weight
        return torch.matmul(u, weight)


class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config):
        vocab_size, process_group, padding_idx = (
            config.vocab_size,
            config.get("process_group", None),
            config.get("padding_idx", None),
        )
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if vocab_size % world_size != 0:
                raise ValueError(
                    f"vocab_size ({vocab_size}) must be divisible by " f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(
            vocab_size // world_size,
            embedding_dim=config.hidden_size,
            padding_idx=padding_idx,
        )

    def embed(self, x: Tensor) -> Tensor:
        if self.process_group is None:
            return self.forward(x)
        rank = torch.distributed.get_rank(self.process_group)
        vocab_size = self.num_embeddings
        vocab_start_index, vocab_end_index = (
            rank * vocab_size,
            (rank + 1) * vocab_size,
        )
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        input_ids_mask = (x < vocab_start_index) | (x >= vocab_end_index)
        x = x - vocab_start_index
        x[input_ids_mask] = 0
        embeddings = self.forward(x)
        embeddings[input_ids_mask] = 0.0
        # Reduce to the global process group
        torch.distributed.all_reduce(embeddings, group=self.process_group)
        return embeddings

    def unembed(self, u: Tensor) -> Tensor:
        if self.process_group is None:
            return u @ self.weight.T
        else:
            raise NotImplementedError
