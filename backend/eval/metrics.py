"""Retrieval evaluation metrics.

All metrics operate on:
    retrieved: list[str] — ranked parent IDs from the retrieval pipeline
    relevant:  set[str]  — ground-truth relevant parent IDs
"""

from __future__ import annotations

import math
import logging
from typing import Sequence

logger = logging.getLogger("eval.metrics")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Recall@K = |relevant ∩ retrieved[:K]| / |relevant|."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Precision@K = |relevant ∩ retrieved[:K]| / K."""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k


def f1_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """F1@K = harmonic mean of precision@K and recall@K."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant doc.

    Returns 0 if no relevant document is found.
    """
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    dcg = 0.0
    idcg = 0.0

    # DCG
    for i, pid in enumerate(retrieved[:k], start=1):
        rel = 1.0 if pid in relevant else 0.0
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    # IDCG (ideal — all relevant documents at top)
    n_rel = min(len(relevant), k)
    for i in range(1, n_rel + 1):
        idcg += 1.0 / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def compute_all_retrieval_metrics(
    retrieved: Sequence[str],
    relevant: set[str],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Compute all retrieval metrics at once.

    Returns a dict like::

        {"recall@1": ..., "recall@3": ..., "mrr": ..., "ndcg@3": ...}
    """
    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"f1@{k}"] = f1_at_k(retrieved, relevant, k)
        if k <= 5:
            result[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    result["mrr"] = mrr(retrieved, relevant)
    return result
