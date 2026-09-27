"""Configuration and tokenization shared by both model backends."""

from mini_infer.common.config import ModelConfig
from mini_infer.common.tokenizer import ByteTokenizer

__all__ = ["ByteTokenizer", "ModelConfig"]

