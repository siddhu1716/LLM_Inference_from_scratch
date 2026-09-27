"""Validation and response helpers for the tiny OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    prompt: str
    max_tokens: int = 32
    temperature: float = 1.0
    stream: bool = False

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CompletionRequest:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        max_tokens = payload.get("max_tokens", 32)
        temperature = payload.get("temperature", 1.0)
        stream = payload.get("stream", False)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise ValueError("'max_tokens' must be an integer in [1, 4096]")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("'temperature' must be a non-negative number")
        if not isinstance(stream, bool):
            raise ValueError("'stream' must be a boolean")
        return cls(prompt, max_tokens, float(temperature), stream)


@dataclass(frozen=True, slots=True)
class ChatCompletionRequest:
    messages: tuple[dict[str, str], ...]
    max_tokens: int = 32
    temperature: float = 1.0
    stream: bool = False

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ChatCompletionRequest:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' must be a non-empty list")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each chat message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError("each chat message needs a valid role and string content")
            normalized.append({"role": role, "content": content})
        completion = CompletionRequest.from_json(
            {
                "prompt": "",
                "max_tokens": payload.get("max_tokens", 32),
                "temperature": payload.get("temperature", 1.0),
                "stream": payload.get("stream", False),
            }
        )
        return cls(
            tuple(normalized),
            completion.max_tokens,
            completion.temperature,
            completion.stream,
        )


def completion_response(model: str, text: str, prompt_tokens: int, output_tokens: int) -> dict:
    """Return the familiar subset of the OpenAI completions response."""
    return {
        "id": f"cmpl-{uuid4().hex}",
        "object": "text_completion",
        "created": int(time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
    }


def chat_completion_response(
    model: str,
    text: str,
    prompt_tokens: int,
    output_tokens: int,
) -> dict:
    response = completion_response(model, text, prompt_tokens, output_tokens)
    response["id"] = response["id"].replace("cmpl-", "chatcmpl-", 1)
    response["object"] = "chat.completion"
    response["choices"][0] = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "length",
    }
    return response
