"""KV-cache memory estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVCacheEstimate:
    bytes_per_token_per_sequence: int
    total_bytes: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024**3)


def _bytes_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    values = (num_layers, num_kv_heads, head_dim, bytes_per_element)
    if any(value <= 0 for value in values):
        raise ValueError("KV-cache dimensions and element size must be positive")
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_element


def estimate_kv_cache(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> KVCacheEstimate:
    if sequence_length < 0 or batch_size <= 0:
        raise ValueError("sequence_length must be non-negative and batch_size must be positive")
    per_token = _bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_element)
    return KVCacheEstimate(per_token, per_token * sequence_length * batch_size)


def max_kv_cache_tokens(
    memory_bytes: int,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    if memory_bytes < 0 or batch_size <= 0:
        raise ValueError("memory_bytes must be non-negative and batch_size must be positive")
    per_token = _bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_element)
    return memory_bytes // (per_token * batch_size)
