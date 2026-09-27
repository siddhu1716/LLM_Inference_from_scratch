"""Stop-token and stop-sequence suffix matching."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopMatch:
    """Description of a matched stop condition."""

    kind: str
    token_ids: tuple[int, ...]


def matches_stop_token(token_id: int, stop_token_ids: list[int]) -> bool:
    """Return whether ``token_id`` is one of the configured stop tokens."""
    _validate_token_id(token_id, "token_id")
    _validate_token_ids(stop_token_ids, "stop_token_ids")
    return token_id in stop_token_ids


def matches_stop_sequence(token_ids: list[int], stop_sequence: list[int]) -> bool:
    """Return whether generated tokens end with a non-empty stop sequence."""
    _validate_token_ids(token_ids, "token_ids")
    _validate_token_ids(stop_sequence, "stop_sequence")
    if not stop_sequence:
        raise ValueError("stop_sequence must not be empty")
    return (
        len(token_ids) >= len(stop_sequence)
        and token_ids[-len(stop_sequence) :] == stop_sequence
    )


def find_stop_match(
    token_ids: list[int],
    *,
    stop_token_ids: list[int] | None = None,
    stop_sequences: list[list[int]] | None = None,
) -> StopMatch | None:
    """Return the longest matching sequence, or a matching final stop token."""
    _validate_token_ids(token_ids, "token_ids")
    configured_tokens = [] if stop_token_ids is None else stop_token_ids
    configured_sequences = [] if stop_sequences is None else stop_sequences
    _validate_token_ids(configured_tokens, "stop_token_ids")
    if not isinstance(configured_sequences, list):
        raise TypeError("stop_sequences must be a list")

    matching_sequences: list[tuple[int, ...]] = []
    for index, sequence in enumerate(configured_sequences):
        _validate_token_ids(sequence, f"stop_sequences[{index}]")
        if not sequence:
            raise ValueError(f"stop_sequences[{index}] must not be empty")
        if len(token_ids) >= len(sequence) and token_ids[-len(sequence) :] == sequence:
            matching_sequences.append(tuple(sequence))
    if matching_sequences:
        matched = max(matching_sequences, key=lambda sequence: (len(sequence), sequence))
        return StopMatch("sequence", matched)
    if token_ids and token_ids[-1] in configured_tokens:
        return StopMatch("token", (token_ids[-1],))
    return None


def should_stop(
    token_ids: list[int],
    *,
    stop_token_ids: list[int] | None = None,
    stop_sequences: list[list[int]] | None = None,
) -> bool:
    """Return whether generated tokens satisfy any configured stop condition."""
    return (
        find_stop_match(
            token_ids,
            stop_token_ids=stop_token_ids,
            stop_sequences=stop_sequences,
        )
        is not None
    )


def _validate_token_ids(token_ids: list[int], name: str) -> None:
    if not isinstance(token_ids, list):
        raise TypeError(f"{name} must be a list")
    for index, token_id in enumerate(token_ids):
        _validate_token_id(token_id, f"{name}[{index}]")


def _validate_token_id(token_id: int, name: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{name} must be an integer")
    if token_id < 0:
        raise ValueError(f"{name} must be non-negative")
