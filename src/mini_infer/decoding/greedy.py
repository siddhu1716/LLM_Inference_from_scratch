"""Greedy token selection."""

from __future__ import annotations

from mini_infer.decoding.logits import argmax


def greedy_decode(logits: list[float]) -> int:
    """Select the highest-logit token, preferring the lowest index on ties."""
    return argmax(logits)
