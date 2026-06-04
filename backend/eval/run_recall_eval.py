#!/usr/bin/env python3
"""Retrieval recall evaluation — measure how well the pipeline retrieves recipes.

Usage
-----
    python -m eval.run_recall_eval [--dataset path] [--output path]

This will:
1. Load the evaluation dataset from ``eval/dataset.json``.
2. For each query, run the full retrieval pipeline.
3. Compute Recall@K, Precision@K, MRR, NDCG.
4. Print summary and save detailed results.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter

# Ensure backend is importable
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import Settings
from app.core.rag.engine import RAGService
from eval.dataset import load_dataset
from eval.metrics import compute_all_retrieval_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.recall")

# Suppress verbose LangChain/httpx logs
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


def run_eval(
    dataset_path: str = None,
    output_path: str = None,
    settings: Settings = None,
) -> list[dict]:
    """Run retrieval evaluation and print results.

    Returns a list of per-item result dicts.
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results_recall.json")

    # 1. Load dataset
    dataset = load_dataset(dataset_path)
    logger.info("Running evaluation on %d queries …", len(dataset))

    # 2. Initialise RAG service (no ingestion — just load persisted parents)
    if settings is None:
        settings = Settings()
    svc = RAGService(settings=settings)
    svc.initialize()
    logger.info("RAGService ready with %d parent recipes", len(svc._ingestion.parent_recipes))

    # 3. Run retrieval for each query
    results: list[dict] = []
    all_metrics_sum: dict[str, float] = {}
    n_errors = 0

    for item in dataset:
        qid = item["id"]
        query = item["query"]
        relevant = set(item["relevant_parent_ids"])

        try:
            t0 = time.time()
            retrieved = svc.retrieve(query)
            elapsed = time.time() - t0

            retrieved_ids = [r.parent_id for r in retrieved]
            metrics = compute_all_retrieval_metrics(retrieved_ids, relevant)

            for k, v in metrics.items():
                all_metrics_sum[k] = all_metrics_sum.get(k, 0.0) + v

            results.append({
                "id": qid,
                "query": query,
                "relevant_count": len(relevant),
                "retrieved_count": len(retrieved),
                "retrieved_ids": retrieved_ids,
                "metrics": metrics,
                "elapsed": round(elapsed, 3),
            })

            if (len(results) + 1) % 100 == 0:
                logger.info("  Progress: %d / %d", len(results), len(dataset))

        except Exception as exc:
            logger.warning("  Error on query '%s': %s", qid, exc)
            n_errors += 1
            results.append({
                "id": qid,
                "query": query,
                "error": str(exc),
            })

    # 4. Compute averages
    n_valid = len(dataset) - n_errors
    avg_metrics = {}
    if n_valid > 0:
        for k in sorted(all_metrics_sum.keys()):
            avg_metrics[k] = round(all_metrics_sum[k] / n_valid, 4)

    # 5. Print summary
    print()
    print("=" * 52)
    print("  RETRIEVAL EVALUATION SUMMARY")
    print("=" * 52)
    print(f"  Dataset size:     {len(dataset)}")
    print(f"  Valid queries:    {n_valid}")
    print(f"  Errors:           {n_errors}")
    print(f"  Total parents:    {len(svc._ingestion.parent_recipes)}")
    print()
    print("  Metric       Value")
    print("  " + "-" * 30)
    for k, v in avg_metrics.items():
        print(f"  {k:<12}  {v:.4f}")
    print("=" * 52)

    # 6. Per-query-type breakdown
    type_groups: dict[str, list[dict]] = {}
    for item in dataset:
        qt = item.get("query_type", "unknown")
        type_groups.setdefault(qt, []).append(item)

    if len(type_groups) > 1:
        print()
        print("  By Query Type:")
        print("  " + "-" * 46)
        for qtype, items in sorted(type_groups.items()):
            group_metrics: dict[str, float] = {}
            group_n = 0
            for r in results:
                if r["id"] in {i["id"] for i in items} and "metrics" in r:
                    for k, v in r["metrics"].items():
                        group_metrics[k] = group_metrics.get(k, 0.0) + v
                    group_n += 1
            if group_n > 0:
                print(f"  [{qtype}] n={group_n}")
                for k in sorted(group_metrics.keys()):
                    print(f"    {k:<12}  {group_metrics[k]/group_n:.4f}")

    # 7. Save detailed results
    output = {
        "config": {
            "dataset": dataset_path,
            "n_parents": len(svc._ingestion.parent_recipes),
            "n_queries": len(dataset),
        },
        "summary": avg_metrics,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info("Detailed results saved to %s", output_path)

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG retrieval recall evaluation")
    parser.add_argument("--dataset", default=None, help="Path to dataset JSON")
    parser.add_argument("--output", default=None, help="Path to output results JSON")
    args = parser.parse_args()

    run_eval(dataset_path=args.dataset, output_path=args.output)


if __name__ == "__main__":
    main()
