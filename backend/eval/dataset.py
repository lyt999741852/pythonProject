"""Evaluation dataset loading and building utilities.

Data format
-----------
Each item in the dataset is an ``EvalItem`` dict with:
    id          — unique query identifier
    query       — natural-language user query
    relevant_parent_ids  — list of parent IDs that should be retrieved
    cuisine     — cuisine category (optional, for grouping)
    query_type  — "specific" (one exact recipe), "recommend" (multiple),
                   "technique" (cooking skill), "general" (free chat)

File format (JSON)
------------------
A JSON list of ``EvalItem`` objects.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger("eval.dataset")

EVAL_ITEM_KEYS = {"id", "query", "relevant_parent_ids", "cuisine", "query_type"}

EvalItem = dict[str, Any]
"""Type alias for a single evaluation item."""


def load_dataset(path: str) -> list[EvalItem]:
    """Load a dataset JSON file and validate its structure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")

    for item in data:
        missing = EVAL_ITEM_KEYS - set(item.keys())
        if missing:
            raise ValueError(f"Item {item.get('id', '?')} missing keys: {missing}")
        if not isinstance(item["relevant_parent_ids"], list):
            raise ValueError(f"Item {item['id']}: relevant_parent_ids must be a list")

    logger.info("Loaded %d evaluation items from %s", len(data), path)
    return data


def save_dataset(dataset: list[EvalItem], path: str) -> None:
    """Save dataset to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d evaluation items to %s", len(dataset), path)


# ---------------------------------------------------------------------------
# Dataset builder — generates evaluation queries from parent recipe data
# ---------------------------------------------------------------------------

QUERY_TEMPLATES_SPECIFIC = [
    "怎么做{name}？",
    "{name}需要什么食材？",
    "{name}的做法是什么？",
    "告诉我{name}的步骤",
    "{name}怎么做好吃？",
    "我想学做{name}，有什么需要注意的？",
]

QUERY_TEMPLATES_RECOMMEND_CUISINE = [
    "推荐几道{cuisine}",
    "有什么好吃的{cuisine}推荐吗？",
    "我想吃点{cuisine}，有什么推荐？",
]

DIFFICULTY_QUERIES: dict[str, list[str]] = {
    "简单": ["推荐几道简单的家常菜", "有没有快手菜？简单好做的"],
    "中等": ["推荐几道中等难度的菜"],
}


def build_dataset_from_recipes(
    parent_recipes: dict[str, dict],
    *,
    n_specific: int = 3,
    n_recommend: int = 2,
    random_seed: int = 42,
) -> list[EvalItem]:
    """Build an evaluation dataset from the parent recipe dict.

    Parameters
    ----------
    parent_recipes:
        The deserialized ``parents.json`` dict: ``{parent_id: RecipeParent_dict}``.
    n_specific:
        Number of single-recipe queries to generate per recipe (sampled without
        replacement from template pool).
    n_recommend:
        Number of "recommend" queries to generate per cuisine type.

    Returns
    -------
    A list of ``EvalItem`` ready to save with ``save_dataset()``.
    """
    items: list[EvalItem] = []
    idx = 0

    # 1. Specific recipe queries (each → a single relevant parent)
    names_seen: set[str] = set()
    for pid, parent in parent_recipes.items():
        name = parent.get("recipe_name", "")
        if not name or name in names_seen:
            continue
        names_seen.add(name)
        cuisine = parent.get("raw_data", {}).get("cuisine_type") or "未知"

        templates = random.Random(f"{random_seed}-{pid}").sample(
            QUERY_TEMPLATES_SPECIFIC, min(n_specific, len(QUERY_TEMPLATES_SPECIFIC))
        )
        for tmpl in templates:
            idx += 1
            items.append({
                "id": f"specific-{idx:04d}",
                "query": tmpl.format(name=name),
                "relevant_parent_ids": [pid],
                "cuisine": cuisine,
                "query_type": "specific",
            })

    # 2. Cross-recipe recommendation queries (per cuisine type)
    cuisine_groups: dict[str, list[str]] = {}
    for pid, parent in parent_recipes.items():
        cuisine = parent.get("raw_data", {}).get("cuisine_type") or "未知"
        cuisine_groups.setdefault(cuisine, []).append(pid)

    for cuisine, pids in cuisine_groups.items():
        if cuisine == "未知":
            continue
        if len(pids) < 2:
            continue

        templates = random.Random(f"{random_seed}-cuisine-{cuisine}").sample(
            QUERY_TEMPLATES_RECOMMEND_CUISINE,
            min(n_recommend, len(QUERY_TEMPLATES_RECOMMEND_CUISINE)),
        )
        for tmpl in templates:
            idx += 1
            items.append({
                "id": f"recommend-{idx:04d}",
                "query": tmpl.format(cuisine=cuisine),
                "relevant_parent_ids": random.Random(f"{random_seed}-q{idx}").sample(
                    pids, min(3, len(pids))
                ),
                "cuisine": cuisine,
                "query_type": "recommend",
            })

    # 3. Difficulty-based queries
    for diff, queries in DIFFICULTY_QUERIES.items():
        matching_pids = [
            pid for pid, parent in parent_recipes.items()
            if parent.get("raw_data", {}).get("difficulty") == diff
        ]
        if matching_pids:
            for query in queries:
                idx += 1
                items.append({
                    "id": f"difficulty-{idx:04d}",
                    "query": query,
                    "relevant_parent_ids": random.Random(f"{random_seed}-{diff}").sample(
                        matching_pids, min(3, len(matching_pids))
                    ),
                    "cuisine": "all",
                    "query_type": "recommend",
                })

    logger.info("Built %d evaluation items from %d parent recipes", len(items), len(parent_recipes))
    return items
