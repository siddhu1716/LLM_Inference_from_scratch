"""Run tiny, inspectable optimization demonstrations."""

from random import Random

from mini_infer.optimizations import (
    dequantize_grouped_int4,
    dequantize_symmetric_int8,
    online_tiled_attention,
    quantize_grouped_int4,
    quantize_symmetric_int8,
    speculative_decode_step,
)

weights = [[-1.0, -0.25, 0.5, 1.0]]
int8 = quantize_symmetric_int8(weights)
print("INT8:", int8, "->", dequantize_symmetric_int8(int8))

int4 = quantize_grouped_int4(weights[0], group_size=2)
print("INT4:", int4, "->", dequantize_grouped_int4(int4))

q = [[1.0, 0.0], [0.0, 1.0]]
print("Online attention:", online_tiled_attention(q, q, q, tile_size=1, causal=True))


def draft_distribution(prefix: tuple[int, ...]) -> dict[int, float]:
    return {1: 0.7, 2: 0.3}


def target_distribution(prefix: tuple[int, ...]) -> dict[int, float]:
    return {1: 0.8, 2: 0.2}


step = speculative_decode_step(
    [0],
    [1, 1],
    draft_distribution,
    target_distribution,
    rng=Random(3),
)
print("Speculative step:", step)
