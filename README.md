# LLM Inference from Scratch

A compact, executable study repository for the algorithms and systems ideas
inside modern LLM inference engines. It intentionally favors readable Python,
small inputs, explicit state, and tests over production throughput.

> **Scope:** this is not a replacement for vLLM, SGLang, TensorRT-LLM, or
> llama.cpp. Hardware-dependent features are implemented as reference math or
> control-plane simulations. There are no custom CUDA kernels, distributed
> collectives, production authentication, or pretrained weights.

## What you can learn here

- Follow a request through admission, prefill, decode, sampling, streaming,
  metrics, and cleanup.
- Compare the same decoder-only Transformer architecture in dependency-free
  Python and `torch.nn`.
- Inspect RMSNorm, RoPE, grouped-query attention, SwiGLU, causal masking, and
  incremental KV caching without a framework hiding the steps.
- Run continuous batching, token-budget and cache-aware scheduling, chunked
  prefill, paged allocation, and longest-prefix reuse on tiny inputs.
- Study executable versions of online-softmax attention, quantization,
  speculative decoding, LoRA, MoE routing, structured decoding, and
  parallelism plans.
- Practice **90 inference-engine interview questions** in the searchable
  GitHub Pages site.

## Feature matrix

| Area | Learning-scale implementation |
|---|---|
| Model | Decoder-only Transformer, embeddings, RMSNorm, RoPE, GQA/MHA, SwiGLU |
| Backends | Pure Python nested lists and PyTorch `torch.nn` |
| KV cache | Transparent per-layer cache plus paged-cache simulation |
| Decoding | Greedy, temperature, categorical, top-k, top-p, min-p, typical |
| Logit controls | Bias, repetition, frequency, and presence penalties |
| Search/stopping | Beam search, stop tokens, and stop sequences |
| Request engine | Lifecycle, prefill/decode phases, token budgets, continuous batching |
| Memory | Block allocator, paged KV cache, radix-style longest-prefix cache |
| Scheduling | Chunked prefill and cache-aware ordering |
| Observability | Streaming events, TTFT, ITL, and throughput metrics |
| Attention optimization | Exact FlashAttention-style tiled online softmax reference |
| Quantization | Symmetric weight-only INT8 and grouped packed INT4 |
| Speculation | Exact rejection sampling and an n-gram proposer |
| Adaptation/routing | LoRA linear adapter and top-k MoE routing |
| Scale-out concepts | Tensor, pipeline, data, expert, and context partition plans |
| Serving architecture | Prefill/decode disaggregation router and static-shape graph cache |
| Structured output | Token DFA and a small JSON-object constraint |
| API | `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, SSE |
| Documentation | Static responsive site, concept search, filters, 90 Q&As |

GPTQ, AWQ, FlashAttention GPU kernels, CUDA Graph capture, NCCL collectives,
and RadixAttention kernels are explained but not falsely presented as
production implementations. Their small counterparts expose the core idea and
invariants.

## Quickstart

Python 3.10 or newer is required.

```powershell
Set-Location "C:\path\to\LLM_Inference_from_scratch"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
python -m pytest
```

Run the tiny **untrained** pure-Python model:

```powershell
mini-infer sample "Hello" --max-new-tokens 8 --seed 7
```

The output is random-model bytes, not useful prose; the purpose is to inspect
the complete autoregressive and KV-cache path.

Run the walkthroughs:

```powershell
python examples\compare_backends.py
python examples\engine_walkthrough.py
python examples\optimization_walkthrough.py
```

Start the OpenAI-style demo server:

```powershell
mini-infer serve --port 8000
curl http://127.0.0.1:8000/health
```

## Repository map

```text
src/mini_infer/
|-- common/          # validated model config and reversible byte tokenizer
|-- backends/
|   |-- python/      # matrix loops and dependency-free Transformer
|   `-- torch/       # matching torch.nn Transformer
|-- decoding/        # one sampling/search/penalty technique per file
|-- engine/          # requests, scheduling, batching, caches, metrics
|-- optimizations/   # executable math and systems simulations
|-- serving/         # stdlib HTTP, schemas, chat template, SSE
`-- cli.py
tests/               # deterministic unit and integration tests
examples/            # runnable learning walkthroughs
docs/                # no-build GitHub Pages site and interview bank
```

Each technique has its own file so it is easy to revisit in isolation.

## Suggested learning path

1. **Model math:** `common/config.py`, then the paired backend files in this
   order: embedding, RMSNorm, RoPE, attention, MLP, block, model.
2. **Token selection:** start with `decoding/logits.py`, then greedy and
   categorical sampling before adding filters and penalties.
3. **Token loop:** read `engine/request.py`, `phases.py`, `scheduler.py`,
   `batching.py`, and finally `sync_engine.py`.
4. **Memory:** compare the model's transparent KV cache with
   `block_allocator.py`, `paged_kv_cache.py`, and `prefix_cache.py`.
5. **Optimizations:** run each file in `optimizations/` beside its tests.
6. **Serving:** trace schemas through `http_server.py`, SSE, and metrics.
7. **Review:** use the website's interview filters to explain tradeoffs aloud.

## Validation

```powershell
python -m pytest
ruff check .
```

Tests cover backend shapes and cache equivalence, deterministic decoding,
request-state and allocator invariants, cache matching and eviction,
attention equivalence, quantization error, speculative acceptance/rejection,
parallel plans, structured decoding, and HTTP behavior.

## Documentation and GitHub Pages

Serve the site locally:

```powershell
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. For GitHub Pages, select **GitHub Actions**
as the Pages source; `.github/workflows/pages.yml` publishes `docs/` directly
with no build dependencies.

## Production-engine mapping

This project takes inspiration from:

- [vLLM documentation](https://docs.vllm.ai/) and the
  [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [SGLang documentation](https://docs.sglang.ai/) and the
  [SGLang paper](https://arxiv.org/abs/2312.07104)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [GPTQ](https://arxiv.org/abs/2210.17323),
  [AWQ](https://arxiv.org/abs/2306.00978), and
  [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)

The names help connect study code to production vocabulary; they do not claim
feature or performance parity.

## License

MIT. See [LICENSE](LICENSE).
