"""A deliberately small chat template used by the HTTP demo."""

from __future__ import annotations

from collections.abc import Iterable


def render_chat(messages: Iterable[dict[str, str]]) -> str:
    """Render role/content messages into an unambiguous educational prompt."""
    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("chat message content must be a string")
        rendered.append(f"<|{role}|>\n{content}\n")
    rendered.append("<|assistant|>\n")
    return "".join(rendered)

