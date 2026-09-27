from __future__ import annotations

import pytest

from mini_infer.backends.python import Transformer as PythonTransformer
from mini_infer.backends.python.matrix import matmul, transpose
from mini_infer.common import ByteTokenizer, ModelConfig


def tiny_config(*, tie_embeddings: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=19,
        dim=8,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        hidden_dim=16,
        max_seq_len=16,
        seed=123,
        tie_embeddings=tie_embeddings,
    )


def test_config_and_byte_tokenizer() -> None:
    tokenizer = ByteTokenizer()
    text = "Transformer café"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == 256

    config = tiny_config()
    assert config.head_dim == 4
    assert config.kv_heads == 1
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(dim=10, n_heads=4)


def test_python_matrix_helpers() -> None:
    assert transpose([[1.0, 2.0], [3.0, 4.0]]) == [[1.0, 3.0], [2.0, 4.0]]
    assert matmul([[1.0, 2.0]], [[3.0], [4.0]]) == [[11.0]]


def test_python_forward_shape_causality_and_determinism() -> None:
    config = tiny_config()
    first = PythonTransformer(config)
    second = PythonTransformer(config)

    logits = first([1, 2, 3, 4])
    changed_future = first([1, 2, 3, 5])
    assert len(logits) == 4
    assert all(len(row) == config.vocab_size for row in logits)
    assert logits == second([1, 2, 3, 4])
    assert logits[:3] == changed_future[:3]


def test_python_incremental_cache_matches_full_forward() -> None:
    config = tiny_config()
    model = PythonTransformer(config)
    tokens = [1, 7, 3, 9]

    full_logits = model(tokens)
    cache = None
    incremental_logits: list[list[float]] = []
    for token in tokens:
        step_logits, cache = model.forward_with_cache([token], cache)
        incremental_logits.extend(step_logits)

    for incremental_row, full_row in zip(incremental_logits, full_logits, strict=True):
        assert incremental_row == pytest.approx(full_row, abs=1e-12)
    assert cache is not None
    assert len(cache) == config.n_layers
    assert all(len(layer_cache) == len(tokens) for layer_cache in cache)
    assert len(cache[0].keys[0]) == config.kv_heads
    assert len(cache[0].keys[0][0]) == config.head_dim


def test_torch_forward_shape_causality_and_determinism() -> None:
    torch = pytest.importorskip("torch")
    from mini_infer.backends.torch import Transformer as TorchTransformer

    config = tiny_config(tie_embeddings=False)
    first = TorchTransformer(config).eval()
    second = TorchTransformer(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    changed_future = torch.tensor([[1, 2, 3, 5]], dtype=torch.long)

    with torch.no_grad():
        logits = first(tokens)
        repeated = second(tokens)
        future_logits = first(changed_future)

    assert logits.shape == (1, 4, config.vocab_size)
    torch.testing.assert_close(logits, repeated, rtol=0.0, atol=0.0)
    torch.testing.assert_close(logits[:, :3], future_logits[:, :3], rtol=0.0, atol=0.0)


def test_torch_incremental_cache_matches_full_forward() -> None:
    torch = pytest.importorskip("torch")
    from mini_infer.backends.torch import Transformer as TorchTransformer

    config = tiny_config()
    model = TorchTransformer(config).eval()
    tokens = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(tokens)
        cache = None
        steps = []
        for position in range(tokens.shape[1]):
            logits, cache = model.forward_with_cache(tokens[:, position : position + 1], cache)
            steps.append(logits)
        incremental_logits = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental_logits, full_logits, rtol=1e-5, atol=1e-6)
    assert cache is not None
    assert len(cache) == config.n_layers
    assert cache[0].keys.shape == (
        tokens.shape[0],
        config.kv_heads,
        tokens.shape[1],
        config.head_dim,
    )
