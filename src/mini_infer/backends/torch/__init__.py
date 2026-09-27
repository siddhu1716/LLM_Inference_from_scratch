"""PyTorch decoder-only Transformer backend."""

try:
    from mini_infer.backends.torch.attention import KVCache
    from mini_infer.backends.torch.model import Transformer
except ImportError as exc:
    if exc.name != "torch":
        raise
    KVCache = None  # type: ignore[assignment,misc]
    Transformer = None  # type: ignore[assignment]

__all__ = ["KVCache", "Transformer"]
