"""A tiny stdlib HTTP server exposing an OpenAI-style completions endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mini_infer.serving.chat_template import render_chat
from mini_infer.serving.openai_schema import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_completion_response,
    completion_response,
)

Generate = Callable[[str, int, float], list[str]]


def make_handler(
    generate: Generate,
    model_name: str = "mini-infer",
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler around a token-producing callback."""

    class CompletionHandler(BaseHTTPRequestHandler):
        server_version = "mini-infer/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok", "model": model_name})
                return
            if self.path == "/v1/models":
                self._json(
                    HTTPStatus.OK,
                    {"object": "list", "data": [{"id": model_name, "object": "model"}]},
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})

        def do_POST(self) -> None:
            if self.path not in {"/v1/completions", "/v1/chat/completions"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})
                return
            try:
                payload = self._read_json()
                is_chat = self.path == "/v1/chat/completions"
                if is_chat:
                    request = ChatCompletionRequest.from_json(payload)
                    prompt = render_chat(request.messages)
                else:
                    request = CompletionRequest.from_json(payload)
                    prompt = request.prompt
                pieces = generate(prompt, request.max_tokens, request.temperature)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                return

            if request.stream:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for piece in pieces:
                    event = {
                        "object": "chat.completion.chunk" if is_chat else "text_completion.chunk",
                        "model": model_name,
                        "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return

            response_factory = chat_completion_response if is_chat else completion_response
            response = response_factory(
                model_name, "".join(pieces), len(prompt.encode()), len(pieces)
            )
            self._json(HTTPStatus.OK, response)

        def _read_json(self) -> dict[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("Content-Length is required")
            length = int(content_length)
            if length > 1_000_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return CompletionHandler


def serve(
    generate: Generate,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "mini-infer",
) -> None:
    """Serve until interrupted."""
    server = ThreadingHTTPServer((host, port), make_handler(generate, model_name))
    print(f"Serving {model_name} at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
