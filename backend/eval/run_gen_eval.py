#!/usr/bin/env python3
"""Generation quality evaluation — LLM-as-Judge.

Evaluates the quality of generated answers along four dimensions:
    Faithfulness   — Is the answer faithful to retrieved context?
    Relevance      — Does the answer address the user's question?
    Completeness   — Does the answer cover all needed information?
    Conciseness    — Is the answer concise and well-structured?

Usage
-----
    python -m eval.run_gen_eval [--dataset path] [--output path] [--n 20]

This will:
1. Load ``n`` queries from the evaluation dataset.
2. For each query, run the full RAG pipeline to generate an answer.
3. Have an LLM judge score each answer on 4 dimensions (1-5 scale).
4. Print summary and save detailed results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time

# Ensure backend is importable
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import Settings
from app.core.rag.config import RAGConfig
from app.core.rag.engine import RAGService
from app.core.rag.llm import create_llm
from eval.dataset import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.gen")

# Suppress verbose libs
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ── Judge prompt ─────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是一位专业的RAG系统评测专家。你的任务是评估AI助手的回答质量。

请从以下四个维度对回答进行评分（1-5分）：

1. **Faithfulness（忠实度）**：回答是否基于给定的参考上下文，有没有编造不存在的信息。
   5 = 完全基于上下文，无任何编造
   4 = 基本基于上下文，有少量合理推断
   3 = 部分基于上下文，但有一些推断或编造
   2 = 大量编造或与上下文矛盾
   1 = 完全脱离上下文

2. **Relevance（相关性）**：回答是否直接针对用户的问题。
   5 = 直接回答问题，完全切题
   4 = 基本切题，略有偏离
   3 = 部分相关，有较多无关内容
   2 = 偏离主题
   1 = 完全不相关

3. **Completeness（完整性）**：回答是否覆盖了问题需要的所有信息。
   5 = 完整覆盖，且信息充分
   4 = 覆盖了大部分，只有少量遗漏
   3 = 覆盖部分，有明显遗漏
   2 = 只覆盖了很少信息
   1 = 没有提供有用信息

4. **Conciseness（简洁性）**：回答是否简洁明了，没有冗余。
   5 = 非常简洁，每个句子都有价值
   4 = 比较简洁，少量冗余
   3 = 有一定冗余但不影响理解
   2 = 明显冗余或过于简略
   1 = 冗长或过于简短

请按以下格式输出评分，每行一项，不要输出其他内容：
忠实度: 5
相关性: 5
完整性: 4
简洁性: 4
理由: 简短理由
"""

JUDGE_HUMAN_TEMPLATE = """## 用户问题
{query}

## 参考上下文（检索到的菜谱信息）
{context}

## AI助手的回答
{answer}
"""


async def evaluate_single(
    query: str,
    answer: str,
    context: str,
    judge_llm,
) -> dict:
    """Use the judge LLM to score a single (query, context, answer) triplet."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", JUDGE_HUMAN_TEMPLATE),
    ])
    messages = prompt.format_messages(query=query, context=context, answer=answer)

    try:
        scores_text = ""
        async for chunk in judge_llm.astream(messages):
            if chunk.content:
                scores_text += chunk.content

        # Regex-based parsing: find "ChineseKey: number" patterns anywhere in text
        scores = {"faithfulness": 0, "relevance": 0, "completeness": 0, "conciseness": 0, "reason": ""}
        key_map = {"忠实度": "faithfulness", "相关性": "relevance",
                   "完整性": "completeness", "简洁性": "conciseness"}

        for zh_key, en_key in key_map.items():
            m = re.search(re.escape(zh_key) + r"\s*[:：]\s*(\d+)", scores_text)
            if m:
                scores[en_key] = int(m.group(1))

        # Reason captures everything after "理由" marker
        m = re.search( r"理由\s*[:：]\s*(.+)", scores_text, re.DOTALL)
        if m:
            scores["reason"] = m.group(1).strip()[:200]

        return scores
    except Exception as exc:
        logger.warning("Judge LLM evaluation failed: %s", exc)
        logger.warning("Raw response: %s", scores_text if 'scores_text' in dir() else "N/A")
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "conciseness": 0, "reason": f"Judge error: {exc}"}


async def run_gen_eval(
    dataset_path: str = None,
    output_path: str = None,
    n_queries: int = 20,
) -> list[dict]:
    """Run generation quality evaluation."""
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results_gen.json")

    dataset = load_dataset(dataset_path)

    # Use specific queries only (they have clear relevance judgments)
    specific_queries = [d for d in dataset if d["query_type"] == "specific"]
    eval_queries = specific_queries[:n_queries]
    logger.info("Evaluating %d queries (out of %d available specific queries) …",
                len(eval_queries), len(specific_queries))

    # ── Initialise RAG service ──
    settings = Settings()
    svc = RAGService(settings=settings)
    svc.initialize()
    logger.info("RAGService ready with %d parent recipes", len(svc._ingestion.parent_recipes))

    # ── Initialise judge LLM (same Moonshot instance) ──
    config = RAGConfig.from_settings(settings)
    judge_llm = create_llm(config)
    logger.info("Judge LLM initialised: %s", config.llm_model)

    # ── Run evaluation ──
    results: list[dict] = []
    scores_sum = {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "conciseness": 0.0}

    for i, item in enumerate(eval_queries):
        qid = item["id"]
        query = item["query"]

        logger.info("[%d/%d] Evaluating: %s", i + 1, len(eval_queries), qid)

        try:
            # Retrieve
            retrieved = svc.retrieve(query)
            parent_ids = [r.parent_id for r in retrieved]

            # Build context
            parents = svc._ingestion.parent_recipes
            context_parts = []
            for pid in parent_ids:
                parent = parents.get(pid)
                if parent:
                    context_parts.append(f"[菜名] {parent.recipe_name}\n{parent.full_text[:500]}")
            context_str = "\n\n".join(context_parts) if context_parts else "(无检索结果)"
            retrieved_names = []
            for pid in parent_ids:
                p = parents.get(pid)
                retrieved_names.append(p.recipe_name if p else "")

            # Generate answer
            answer_tokens: list[str] = []
            async for token in svc.process_query(query):
                answer_tokens.append(token)
            answer = "".join(answer_tokens)

            # Judge
            scores = await evaluate_single(query, answer, context_str, judge_llm)

            for dim in scores_sum:
                scores_sum[dim] += scores.get(dim, 0)

            results.append({
                "id": qid,
                "query": query,
                "retrieved_recipe_names": retrieved_names,
                "answer": answer,
                "context_length": len(context_str),
                "scores": scores,
            })

            logger.info("  → F={} R={} C={} Con={}".format(
                scores.get("faithfulness", "?"),
                scores.get("relevance", "?"),
                scores.get("completeness", "?"),
                scores.get("conciseness", "?"),
            ))

        except Exception as exc:
            logger.error("  Error on '%s': %s", qid, exc)
            results.append({"id": qid, "query": query, "error": str(exc)})

    # ── Summary ──
    n_valid = len([r for r in results if "scores" in r])
    print()
    print("=" * 50)
    print("  GENERATION QUALITY EVALUATION")
    print("=" * 50)
    print(f"  Evaluated: {n_valid} queries")
    print()
    if n_valid > 0:
        print("  Dimension       Avg Score")
        print("  " + "-" * 30)
        for dim in ["faithfulness", "relevance", "completeness", "conciseness"]:
            avg = scores_sum[dim] / n_valid if n_valid > 0 else 0
            print(f"  {dim:<14}  {avg:.2f} / 5.0")
    print("=" * 50)

    # Save
    output = {
        "config": {
            "dataset": dataset_path,
            "n_parents": len(svc._ingestion.parent_recipes),
            "judge_model": config.llm_model,
        },
        "summary": {dim: round(scores_sum[dim] / n_valid, 4) for dim in scores_sum} if n_valid > 0 else {},
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info("Detailed results saved to %s", output_path)

    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG generation quality evaluation")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=20, help="Number of queries to evaluate")
    args = parser.parse_args()

    asyncio.run(run_gen_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        n_queries=args.n,
    ))


if __name__ == "__main__":
    main()
