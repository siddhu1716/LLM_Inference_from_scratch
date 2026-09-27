"""Structured token generation with a deterministic finite automaton."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenDFA:
    start_state: str
    accepting_states: frozenset[str]
    transitions: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if self.start_state not in self.transitions:
            raise ValueError("start_state must have a transition entry")
        known = set(self.transitions)
        if not self.accepting_states.issubset(known):
            raise ValueError("accepting states must be declared in transitions")
        for outgoing in self.transitions.values():
            if any(target not in known for target in outgoing.values()):
                raise ValueError("transition targets must be declared states")


class StructuredDecoder:
    def __init__(self, dfa: TokenDFA) -> None:
        self.dfa = dfa
        self.state = dfa.start_state

    @property
    def allowed_tokens(self) -> frozenset[str]:
        return frozenset(self.dfa.transitions[self.state])

    @property
    def is_accepting(self) -> bool:
        return self.state in self.dfa.accepting_states

    def consume(self, token: str) -> str:
        try:
            self.state = self.dfa.transitions[self.state][token]
        except KeyError as exc:
            raise ValueError(f"token {token!r} is not allowed in state {self.state!r}") from exc
        return self.state


def json_object_dfa() -> TokenDFA:
    """Return a token-level DFA for a flat JSON-like object."""
    return TokenDFA(
        start_state="start",
        accepting_states=frozenset({"done"}),
        transitions={
            "start": {"{": "first_key_or_end"},
            "first_key_or_end": {"STRING": "colon", "}": "done"},
            "key": {"STRING": "colon"},
            "colon": {":": "value"},
            "value": {"STRING": "comma_or_end", "NUMBER": "comma_or_end"},
            "comma_or_end": {",": "key", "}": "done"},
            "done": {},
        },
    )
