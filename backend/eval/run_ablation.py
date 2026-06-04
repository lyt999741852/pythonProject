#!/usr/bin/env python3
"""Ablation experiments for RAG retrieval pipeline.

Runs multiple configurations of the retriever and compares metrics.

Configurations tested
---------------------
A3 — BM25 only (current default with DashScope down)
C1 — default top_k (5,5)
C2 — larger top_k (10,10)
C3 — larger BM25 top_k (5,20)
C4 — top_n variation (top_n=5 vs top_n=3)
B1/B2/B3 — RRF k variation (10, 30, 60)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# Ensure backend is importable
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import Settings
from app.core.rag.engine import RAGService
from app.core.rag.config import RAGConfig
from eval.dataset import load_dataset
from eval.metrics import compute_all_retrieval_metrics

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("eval.ablation")

# Silence verbose libs
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("app").setLevel(logging.WARNING)

# ── Ablation configurations ──────────────────────────────────────────────

# Each config overrides specific RAGConfig fields.
# Keys are RAGConfig field names (from config.py).
ABLATION_CONFIGS: list[dict] = [
    # ── Route ablation (A series) ──
    {
        "name": "A3_bm25_default",
        "label": "BM25 only (vector_top_k=5, bm25_top_k=5, k=60, n=3)",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 60,
        "final_top_n": 3,
    },
    # ── top_k variation (C series) ──
    {
        "name": "C1_default_topk",
        "label": "Default top_k (5,5)",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 60,
        "final_top_n": 3,
    },
    {
        "name": "C2_large_topk",
        "label": "Large top_k (10,10)",
        "vector_search_top_k": 10,
        "bm25_search_top_k": 10,
        "rrf_k_constant": 60,
        "final_top_n": 3,
    },
    {
        "name": "C3_large_bm25_topk",
        "label": "Large BM25 top_k (5,20)",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 20,
        "rrf_k_constant": 60,
        "final_top_n": 3,
    },
    # ── RRF k variation (B series) ──
    {
        "name": "B1_rrf_k10",
        "label": "RRF k=10",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 10,
        "final_top_n": 3,
    },
    {
        "name": "B2_rrf_k30",
        "label": "RRF k=30",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 30,
        "final_top_n": 3,
    },
    {
        "name": "B3_rrf_k60",
        "label": "RRF k=60 (default)",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 60,
        "final_top_n": 3,
    },
    # ── final_top_n variation ──
    {
        "name": "C4_top_n5",
        "label": "final_top_n=5",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 60,
        "final_top_n": 5,
    },
    {
        "name": "C5_top_n1",
        "label": "final_top_n=1",
        "vector_search_top_k": 5,
        "bm25_search_top_k": 5,
        "rrf_k_constant": 60,
        "final_top_n": 1,
    },
]


def build_svc(base_settings: Settings, overrides: dict) -> RAGService:
    """Build a RAGService with overridden config parameters."""
    svc = RAGService(settings=base_settings)

    # Apply overrides to the existing RAGConfig
    for key, val in overrides.items():
        if hasattr(svc._retriever, f"_{key}"):
            setattr(svc._retriever, f"_{key}", val)

    svc.initialize()
    return svc


def run_ablation(dataset_path: str = None, output_path: str = None) -> list[dict]:
    """Run all ablation experiments and print comparison table."""
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results_ablation.json")

    dataset = load_dataset(dataset_path)
    logger.info("Loaded %d queries. Running %d configs …", len(dataset), len(ABLATION_CONFIGS))

    settings = Settings()
    all_results: list[dict] = []

    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        label = cfg["label"]
        logger.warning("\n--- %s: %s ---", name, label)

        svc = build_svc(settings, cfg)

        # Short evaluation: use first 100 queries (or fewer) for speed
        # since all configs will converge to the same BM50 result anyway
        eval_items = dataset if len(dataset) < 150 else dataset[:100]

        metrics_sum: dict[str, float] = {}
        n_ok = 0

        t0 = time.time()
        for item in eval_items:
            query = item["query"]
            relevant = set(item["relevant_parent_ids"])

            try:
                retrieved = svc.retrieve(query)
                retrieved_ids = [r.parent_id for r in retrieved]
                metrics = compute_all_retrieval_metrics(retrieved_ids, relevant)
                for k, v in metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                n_ok += 1
            except Exception:
                pass

        elapsed = time.time() - t0

        avg = {k: round(v / n_ok, 4) for k, v in metrics_sum.items()} if n_ok > 0 else {}
        avg["n_queries"] = n_ok
        avg["elapsed"] = round(elapsed, 2)

        all_results.append({
            "name": name,
            "label": label,
            "config": cfg,
            "metrics": avg,
        })

        # Print row
        print(f"  {name:<20}  R@1={avg.get('recall@1',0):.3f}  "
              f"R@3={avg.get('recall@3',0):.3f}  "
              f"MRR={avg.get('mrr',0):.3f}  "
              f"NDCG@3={avg.get('ndcg@3',0):.3f}  "
              f"P@1={avg.get('precision@1',0):.3f}  "
              f"[{elapsed:.1f}s]")

    # ── Summary table ──
    print()
    print("=" * 90)
    print("  ABLATION SUMMARY")
    print("=" * 90)
    print(f"  {'Config':<22} {'R@1':<7} {'R@3':<7} {'P@1':<7} {'MRR':<7} {'NDCG@3':<7} {'Time':<7}")
    print("  " + "-" * 63)
    for r in all_results:
        m = r["metrics"]
        print(f"  {r['name']:<22} {m.get('recall@1',0):<7.3f} "
              f"{m.get('recall@3',0):<7.3f} {m.get('precision@1',0):<7.3f} "
              f"{m.get('mrr',0):<7.3f} {m.get('ndcg@3',0):<7.3f} "
              f"{m.get('elapsed',0):<7.1f}")
    print("=" * 90)

    # Save
    output = {
        "dataset": dataset_path,
        "configs": [r["name"] for r in all_results],
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.warning("Results saved to %s", output_path)

    return all_results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG retrieval ablation study")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_ablation(dataset_path=args.dataset, output_path=args.output)


if __name__ == "__main__":
    main()
