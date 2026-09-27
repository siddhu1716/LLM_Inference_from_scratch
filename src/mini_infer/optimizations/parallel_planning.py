"""Tensor/pipeline/data/expert/context parallel partition planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParallelismConfig:
    tensor: int = 1
    pipeline: int = 1
    data: int = 1
    expert: int = 1
    context: int = 1

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.axes):
            raise ValueError("parallelism degrees must be positive")

    @property
    def axes(self) -> tuple[int, int, int, int, int]:
        return self.data, self.pipeline, self.tensor, self.expert, self.context

    @property
    def world_size(self) -> int:
        result = 1
        for value in self.axes:
            result *= value
        return result


@dataclass(frozen=True, slots=True)
class RankPartition:
    rank: int
    data_rank: int
    pipeline_rank: int
    tensor_rank: int
    expert_rank: int
    context_rank: int
    layer_range: range
    context_range: range | None


@dataclass(frozen=True, slots=True)
class ParallelismPlan:
    config: ParallelismConfig
    ranks: tuple[RankPartition, ...]


def plan_parallelism(
    *,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_experts: int,
    config: ParallelismConfig,
    sequence_length: int | None = None,
) -> ParallelismPlan:
    values = (num_layers, hidden_size, num_attention_heads, num_experts)
    if any(value <= 0 for value in values):
        raise ValueError("model dimensions must be positive")
    checks = (
        (num_layers, config.pipeline, "layers"),
        (hidden_size, config.tensor, "hidden_size"),
        (num_attention_heads, config.tensor, "attention heads"),
        (num_experts, config.expert, "experts"),
    )
    if sequence_length is not None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        checks += ((sequence_length, config.context, "sequence length"),)
    for value, degree, name in checks:
        if value % degree:
            raise ValueError(f"{name} must be divisible by its parallelism degree")
    layers_per_stage = num_layers // config.pipeline
    context_per_rank = (
        None if sequence_length is None else sequence_length // config.context
    )
    ranks: list[RankPartition] = []
    rank = 0
    for data_rank in range(config.data):
        for pipeline_rank in range(config.pipeline):
            for tensor_rank in range(config.tensor):
                for expert_rank in range(config.expert):
                    for context_rank in range(config.context):
                        context_range = (
                            None
                            if context_per_rank is None
                            else range(
                                context_rank * context_per_rank,
                                (context_rank + 1) * context_per_rank,
                            )
                        )
                        ranks.append(
                            RankPartition(
                                rank,
                                data_rank,
                                pipeline_rank,
                                tensor_rank,
                                expert_rank,
                                context_rank,
                                range(
                                    pipeline_rank * layers_per_stage,
                                    (pipeline_rank + 1) * layers_per_stage,
                                ),
                                context_range,
                            )
                        )
                        rank += 1
    return ParallelismPlan(config, tuple(ranks))
