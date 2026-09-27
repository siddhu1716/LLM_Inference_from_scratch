"""Command-line entry points for the educational model and server."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mini_infer.backends.python.model import Transformer
from mini_infer.common.config import ModelConfig
from mini_infer.common.tokenizer import ByteTokenizer
from mini_infer.decoding.pipeline import Sampler, SamplingConfig
from mini_infer.serving.http_server import serve


def _tiny_model(seed: int) -> Transformer:
    return Transformer(
        ModelConfig(
            dim=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            hidden_dim=32,
            max_seq_len=512,
            seed=seed,
        )
    )


def _generate(
    model: Transformer,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    *,
    seed: int,
) -> list[int]:
    sampler = Sampler(
        SamplingConfig(
            temperature=temperature,
            greedy=temperature == 0,
            top_k=32,
            seed=seed,
        )
    )
    logits, cache = model.forward_with_cache(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token_id = sampler.sample(logits[-1], list(prompt_ids) + generated)
        generated.append(token_id)
        logits, cache = model.forward_with_cache([token_id], cache)
    return generated


def _sample(args: argparse.Namespace) -> None:
    tokenizer = ByteTokenizer()
    model = _tiny_model(args.seed)
    prompt_ids = tokenizer.encode(args.prompt)
    generated = _generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.temperature,
        seed=args.seed,
    )
    rendered = tokenizer.decode(generated, errors="backslashreplace")
    print(rendered.encode("unicode_escape").decode("ascii"))
    print(f"\nToken IDs: {generated}")


def _serve(args: argparse.Namespace) -> None:
    tokenizer = ByteTokenizer()
    model = _tiny_model(args.seed)

    def generate(prompt: str, count: int, temperature: float) -> list[str]:
        token_ids = _generate(
            model,
            tokenizer.encode(prompt),
            count,
            temperature,
            seed=args.seed,
        )
        return [bytes([token_id]).decode("utf-8", errors="replace") for token_id in token_ids]

    serve(generate, host=args.host, port=args.port, model_name="mini-infer-random")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-infer",
        description="Run the tiny untrained reference model or its HTTP demo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="sample from the pure-Python model")
    sample.add_argument("prompt")
    sample.add_argument("--max-new-tokens", type=int, default=8)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--seed", type=int, default=0)
    sample.set_defaults(handler=_sample)

    server = subparsers.add_parser("serve", help="start the OpenAI-style demo server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--seed", type=int, default=0)
    server.set_defaults(handler=_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_new_tokens", 1) <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
