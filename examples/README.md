# Examples

- `compare_backends.py` runs the same tiny shape configuration through the
  dependency-free and PyTorch models.
- `engine_walkthrough.py` prints request state transitions for prefill,
  continuous batching, paged allocation, and decode.
- `optimization_walkthrough.py` demonstrates quantization, online-softmax
  attention, and speculative decoding with tiny deterministic inputs.

These examples optimize for readability and observability, not speed.

Run them from the repository root after `python -m pip install -e ".[all]"`:

```powershell
python examples\compare_backends.py
python examples\engine_walkthrough.py
python examples\optimization_walkthrough.py
```
