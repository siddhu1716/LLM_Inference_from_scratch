"""Grouped symmetric int4 quantization with packed nibbles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroupedInt4Weights:
    packed_values: bytes
    scales: tuple[float, ...]
    group_size: int
    value_count: int


def quantize_grouped_int4(
    values: Sequence[float],
    *,
    group_size: int = 32,
) -> GroupedInt4Weights:
    data = tuple(float(value) for value in values)
    if not data:
        raise ValueError("values must not be empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    scales: list[float] = []
    quantized: list[int] = []
    for start in range(0, len(data), group_size):
        group = data[start : start + group_size]
        maximum = max(abs(value) for value in group)
        scale = maximum / 7.0 if maximum else 1.0
        scales.append(scale)
        quantized.extend(max(-7, min(7, round(value / scale))) for value in group)
    packed = bytearray()
    for index in range(0, len(quantized), 2):
        low = quantized[index] & 0xF
        high = quantized[index + 1] & 0xF if index + 1 < len(quantized) else 0
        packed.append(low | (high << 4))
    return GroupedInt4Weights(bytes(packed), tuple(scales), group_size, len(data))


def dequantize_grouped_int4(weights: GroupedInt4Weights) -> tuple[float, ...]:
    if weights.group_size <= 0 or weights.value_count <= 0:
        raise ValueError("invalid grouped int4 metadata")
    unpacked: list[int] = []
    for packed in weights.packed_values:
        for nibble in (packed & 0xF, packed >> 4):
            unpacked.append(nibble - 16 if nibble >= 8 else nibble)
    unpacked = unpacked[: weights.value_count]
    expected_scales = (weights.value_count + weights.group_size - 1) // weights.group_size
    if len(weights.scales) != expected_scales:
        raise ValueError("scale count does not match grouped int4 metadata")
    return tuple(
        value * weights.scales[index // weights.group_size]
        for index, value in enumerate(unpacked)
    )
