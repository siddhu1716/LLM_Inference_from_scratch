"""Compare output shapes and incremental-cache behavior in both backends."""

from mini_infer.backends.python import Transformer as PythonTransformer
from mini_infer.common import ModelConfig


def main() -> None:
    config = ModelConfig(
        vocab_size=256,
        dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        hidden_dim=32,
        seed=7,
    )
    token_ids = [72, 105]

    python_model = PythonTransformer(config)
    python_logits, python_cache = python_model.forward_with_cache(token_ids)
    next_logits, python_cache = python_model.forward_with_cache([33], python_cache)
    print("Pure Python logits:", len(python_logits), "x", len(python_logits[0]))
    print("Pure Python cached tokens:", len(python_cache[0]))
    print("Pure Python next-step rows:", len(next_logits))

    try:
        import torch

        from mini_infer.backends.torch import Transformer as TorchTransformer
    except ImportError:
        print("PyTorch is not installed; install mini-infer[torch] for the tensor path.")
        return

    assert TorchTransformer is not None
    torch_model = TorchTransformer(config).eval()
    with torch.inference_mode():
        torch_logits, torch_cache = torch_model.forward_with_cache(
            torch.tensor([token_ids], dtype=torch.long)
        )
        next_torch_logits, torch_cache = torch_model.forward_with_cache(
            torch.tensor([[33]], dtype=torch.long),
            torch_cache,
        )
    print("PyTorch logits:", tuple(torch_logits.shape))
    print("PyTorch cached tokens:", torch_cache[0].seq_len)
    print("PyTorch next-step shape:", tuple(next_torch_logits.shape))


if __name__ == "__main__":
    main()
