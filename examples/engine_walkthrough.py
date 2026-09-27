"""Observe continuous batching, request phases, events, and metrics."""

from mini_infer.engine import ExecutionPhase, SynchronousEngine


def model(tokens: tuple[int, ...], phase: ExecutionPhase) -> int:
    print(f"{phase.value:7} tokens={tokens}")
    return (tokens[-1] + 1) % 256


engine = SynchronousEngine(model, max_batch_size=2, max_batch_tokens=16)
first = engine.submit([1, 2], max_new_tokens=3, request_id="first")
second = engine.submit([10], max_new_tokens=2, request_id="second")

while engine.batcher.has_work:
    progressed = engine.step()
    print("step:", [(request.request_id, request.state.value) for request in progressed])

print("outputs:", engine.run_until_complete())
print("first events:", engine.events.drain(first.request_id))
print("second events:", engine.events.drain(second.request_id))
print("metrics:", engine.metrics.snapshot())
