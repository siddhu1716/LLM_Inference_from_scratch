from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from mini_infer.serving.chat_template import render_chat
from mini_infer.serving.http_server import make_handler
from mini_infer.serving.openai_schema import ChatCompletionRequest, CompletionRequest


def test_chat_template_validates_roles() -> None:
    assert render_chat([{"role": "user", "content": "hi"}]).endswith("<|assistant|>\n")

    try:
        render_chat([{"role": "tool", "content": "no"}])
    except ValueError as exc:
        assert "unsupported chat role" in str(exc)
    else:
        raise AssertionError("unsupported role was accepted")


def test_completion_request_validation() -> None:
    request = CompletionRequest.from_json({"prompt": "hello", "max_tokens": 2})
    assert request.prompt == "hello"
    assert request.max_tokens == 2
    chat = ChatCompletionRequest.from_json(
        {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert chat.messages[0]["role"] == "user"


def test_http_completion_and_health() -> None:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(lambda prompt, count, temperature: ["x"] * count),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    connection = HTTPConnection(host, port)
    try:
        connection.request("GET", "/health")
        health = connection.getresponse()
        assert health.status == 200

        payload = json.dumps({"prompt": "hi", "max_tokens": 3})
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        assert body["choices"][0]["text"] == "xxx"
        assert body["usage"]["completion_tokens"] == 3

        chat_payload = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=chat_payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(chat_payload)),
            },
        )
        chat_response = connection.getresponse()
        chat_body = json.loads(chat_response.read())
        assert chat_response.status == 200
        assert chat_body["choices"][0]["message"]["content"] == "x"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
