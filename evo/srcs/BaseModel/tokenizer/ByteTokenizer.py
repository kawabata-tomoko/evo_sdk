# based on https://github.com/EleutherAI/gpt-neox/blob/main/megatron/tokenizer/tokenizer.py
from __future__ import annotations

import torch

import numpy as np

from os import PathLike
from typing import List, Tuple, Dict

from tokenizers import Tokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils.generic import TensorType, PaddingStrategy


EMPTY: str = ""


class ByteTokenizer(PreTrainedTokenizer):

    """UTF-8 Encoder."""

    @classmethod
    def from_pretrained(cls, model_id: str | PathLike, **kwargs) -> ByteTokenizer:

        return cls(**kwargs, byte_level=True)

    @property
    def vocab_size(self) -> int:

        return 512

    @property
    def byte_level(self) -> bool:

        return self.init_kwargs.get('byte_level', True)

    def get_vocab(self) -> Dict[str, int]:

        return {chr(i): i for i in range(self.vocab_size)}

    def __len__(self) -> int:

        return self.vocab_size

    def clamp(self, n: int) -> int:

        return max(32, min(n, self.vocab_size))

    def _tokenize(self, text: str, **kwargs) -> List[str]:

        return list(text)

    def byte_tokenize(self, text: str) -> np.ndarray:

        return np.frombuffer(text.encode('utf-8'), dtype=np.uint8)

    def _convert_token_to_id(self, token: str) -> int:

        return self.clamp(ord(token))

    def _convert_id_to_token(self, index: int) -> str:

        return chr(self.clamp(index))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:

        return EMPTY.join(tokens)

    def _decode(self, token_ids: List[int], **kwargs) -> str:

        indices = np.asarray(token_ids, dtype=np.uint8)

        return (
            indices.clip(min=32, max=self.vocab_size, out=indices)
            .tobytes()
            .decode('utf-8')
        )

    def _encode_plus(self, text: str, **kwargs) -> BatchEncoding:

        first_ids = self.byte_tokenize(text).tolist()

        return self.prepare_for_model(
            first_ids,
            pair_ids=None,
            add_special_tokens=kwargs.get('add_special_tokens', False),
            padding=kwargs.get('padding_strategy', PaddingStrategy.DO_NOT_PAD).value,
            truncation=kwargs.get('truncation_strategy', TruncationStrategy.DO_NOT_TRUNCATE).value,
            max_length=kwargs.get('max_length'),
            stride=kwargs.get('stride', 0),
            pad_to_multiple_of=kwargs.get('pad_to_multiple_of'),
            return_tensors=kwargs.get('return_tensors'),
            prepend_batch_axis=True,
            return_attention_mask=kwargs.get('return_attention_mask'),
            return_token_type_ids=kwargs.get('return_token_type_ids'),
            return_overflowing_tokens=kwargs.get('return_overflowing_tokens', False),
            return_special_tokens_mask=kwargs.get('return_special_tokens_mask', False),
            return_length=kwargs.get('return_length', False),
            verbose=kwargs.get('verbose', True),
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs: List[str], **kwargs) -> BatchEncoding:

        input_ids = [(self.byte_tokenize(text).tolist(), None) for text in batch_text_or_text_pairs]

        return self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=kwargs.get('add_special_tokens', False),
            padding_strategy=kwargs.get('padding_strategy', PaddingStrategy.DO_NOT_PAD),
            truncation_strategy=kwargs.get('truncation_strategy', TruncationStrategy.DO_NOT_TRUNCATE),
            max_length=kwargs.get('max_length'),
            stride=kwargs.get('stride', 0),
            pad_to_multiple_of=kwargs.get('pad_to_multiple_of'),
            return_attention_mask=kwargs.get('return_attention_mask'),
            return_token_type_ids=kwargs.get('return_token_type_ids'),
            return_overflowing_tokens=kwargs.get('return_overflowing_tokens', False),
            return_special_tokens_mask=kwargs.get('return_special_tokens_mask', False),
            return_length=kwargs.get('return_length', False),
            return_tensors=kwargs.get('return_tensors'),
            verbose=kwargs.get('verbose', True),
        )

    def _save_pretrained(
        self, save_directory: str | PathLike, file_names: Tuple[str], **kwargs
    ) -> Tuple[str]:

        return file_names
