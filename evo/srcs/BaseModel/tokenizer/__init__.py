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
from .ByteTokenizer import ByteTokenizer