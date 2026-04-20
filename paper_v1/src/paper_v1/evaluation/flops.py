"""Simple FLOPs estimates for the small models used in paper_v1."""

from __future__ import annotations

from typing import Iterable


def linear_layer_flops(input_dim: int, output_dim: int, *, bias: bool = True) -> int:
    return (2 * input_dim * output_dim) + (output_dim if bias else 0)


def mlp_flops(layer_dims: Iterable[int]) -> int:
    dims = list(layer_dims)
    total = 0
    for start, end in zip(dims, dims[1:]):
        total += linear_layer_flops(start, end, bias=True)
    return total


def retrieval_flops(embedding_dim: int, num_prototypes: int, top_k: int) -> dict[str, int]:
    cosine_cost = 2 * embedding_dim * num_prototypes
    topk_cost = num_prototypes
    aggregation_cost = top_k * embedding_dim
    return {
        "cosine_similarity_flops": cosine_cost,
        "topk_selection_flops": topk_cost,
        "aggregation_flops": aggregation_cost,
        "total_retrieval_flops": cosine_cost + topk_cost + aggregation_cost,
    }
