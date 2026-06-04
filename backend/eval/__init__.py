"""RAG evaluation suite — retrieval metrics, ablation studies, generation quality."""

from __future__ import annotations

from .metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    f1_at_k,
    compute_all_retrieval_metrics,
)
from .dataset import (
    EvalItem,
    load_dataset,
    build_dataset_from_recipes,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "f1_at_k",
    "compute_all_retrieval_metrics",
    "EvalItem",
    "load_dataset",
    "build_dataset_from_recipes",
]
