"""Small, executable demonstrations of inference optimizations."""

from mini_infer.optimizations.cuda_graph_simulator import StaticShapeGraphCache
from mini_infer.optimizations.disaggregated_routing import PrefillDecodeRouter, WorkerKind
from mini_infer.optimizations.flash_attention import online_tiled_attention
from mini_infer.optimizations.int4_quantization import (
    GroupedInt4Weights,
    dequantize_grouped_int4,
    quantize_grouped_int4,
)
from mini_infer.optimizations.int8_quantization import (
    SymmetricInt8Weights,
    dequantize_symmetric_int8,
    quantize_symmetric_int8,
)
from mini_infer.optimizations.kv_cache_estimator import (
    KVCacheEstimate,
    estimate_kv_cache,
    max_kv_cache_tokens,
)
from mini_infer.optimizations.lora import LoRAAdapter, lora_linear
from mini_infer.optimizations.moe_routing import ExpertAssignment, top_k_route
from mini_infer.optimizations.ngram_proposer import NGramProposer
from mini_infer.optimizations.parallel_planning import (
    ParallelismConfig,
    ParallelismPlan,
    plan_parallelism,
)
from mini_infer.optimizations.speculative_decoding import (
    SpeculativeStep,
    speculative_decode_step,
)
from mini_infer.optimizations.structured_generation import (
    StructuredDecoder,
    TokenDFA,
    json_object_dfa,
)
from mini_infer.optimizations.weight_loading import load_json_weight_shards
from mini_infer.optimizations.weight_sharding import merge_row_shards, shard_rows

__all__ = [
    "ExpertAssignment",
    "GroupedInt4Weights",
    "KVCacheEstimate",
    "LoRAAdapter",
    "NGramProposer",
    "ParallelismConfig",
    "ParallelismPlan",
    "PrefillDecodeRouter",
    "SpeculativeStep",
    "StaticShapeGraphCache",
    "StructuredDecoder",
    "SymmetricInt8Weights",
    "TokenDFA",
    "WorkerKind",
    "dequantize_grouped_int4",
    "dequantize_symmetric_int8",
    "estimate_kv_cache",
    "json_object_dfa",
    "load_json_weight_shards",
    "lora_linear",
    "max_kv_cache_tokens",
    "merge_row_shards",
    "online_tiled_attention",
    "plan_parallelism",
    "quantize_grouped_int4",
    "quantize_symmetric_int8",
    "shard_rows",
    "speculative_decode_step",
    "top_k_route",
]
