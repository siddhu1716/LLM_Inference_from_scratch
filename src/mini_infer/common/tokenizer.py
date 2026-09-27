"""A reversible dependency-free byte tokenizer."""

from __future__ import annotations

from collections.abc import Iterable


class ByteTokenizer:
    """Map UTF-8 bytes directly to token IDs in ``[0, 256)``.

    This tokenizer has no learned vocabulary and is therefore useful for
    examples and deterministic tests, not for efficient language modeling.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        """Encode text to UTF-8 byte IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Iterable[int], *, errors: str = "strict") -> str:
        """Decode byte IDs, forwarding the standard UTF-8 error policy."""
        values = list(token_ids)
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in values):
            raise TypeError("token IDs must be integers")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in values):
            raise ValueError("token IDs must be in [0, 256)")
        return bytes(values).decode("utf-8", errors=errors)
