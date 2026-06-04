#!/usr/bin/env python3
"""One-shot script to build the evaluation dataset from persisted parent data."""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure backend is importable
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from eval.dataset import build_dataset_from_recipes, save_dataset

DATA_DIR = os.path.join(_backend_dir, "data")
PARENTS_PATH = os.path.join(DATA_DIR, "parents.json")
DATASET_PATH = os.path.join(_backend_dir, "eval", "dataset.json")


def main() -> None:
    if not os.path.isfile(PARENTS_PATH):
        print(f"ERROR: parents.json not found at {PARENTS_PATH}")
        print("Start the backend first so recipes are ingested and persisted.")
        sys.exit(1)

    with open(PARENTS_PATH, "r", encoding="utf-8") as fh:
        parents = json.load(fh)

    print(f"Loaded {len(parents)} parent recipes from {PARENTS_PATH}")
    dataset = build_dataset_from_recipes(parents, n_specific=2, n_recommend=1)

    save_dataset(dataset, DATASET_PATH)

    # Print summary
    specific = [d for d in dataset if d["query_type"] == "specific"]
    recommend = [d for d in dataset if d["query_type"] == "recommend"]
    print(f"\nDataset summary:")
    print(f"  Specific queries:  {len(specific)}")
    print(f"  Recommend queries: {len(recommend)}")
    print(f"  Total:             {len(dataset)}")
    print(f"\nSample queries:")
    for item in dataset[:5]:
        print(f"  [{item['id']}] {item['query']} → {len(item['relevant_parent_ids'])} relevant")


if __name__ == "__main__":
    main()
