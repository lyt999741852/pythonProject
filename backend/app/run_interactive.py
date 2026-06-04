"""
Interactive CLI for testing the RAG pipeline with real recipe data.

Usage:
    pip install -e .             # one-time setup
    python -m app.run_interactive        # from backend/ directory
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# Ensure the ``backend`` directory is on sys.path so ``from app.xxx`` works
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import Settings
from app.recipe_parser import parse_all_recipes
from app.services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_interactive")

# Path to the dishes directory (relative to backend/)
DISHES_DIR = os.path.join(_backend_dir, "dishes")


async def main() -> None:
    """Entry-point: init RAGService, ingest all recipes, enter query loop."""
    # ── 0. Validate dishes directory ──────────────────────────────────────
    if not os.path.isdir(DISHES_DIR):
        print(f"[!] Dishes directory not found: {DISHES_DIR}")
        print(f"    Make sure 'dishes/' exists alongside 'app/' in the backend folder.")
        sys.exit(1)

    # ── 1. Initialise RAGService ──────────────────────────────────────────
    print("=" * 60)
    print("  Vibe Cooking — 智能烹饪助手 交互式测试")
    print("=" * 60)
    print()
    print("[1/4] 初始化 RAGService...")
    settings = Settings()
    svc = RAGService(settings=settings)
    svc.initialize()
    print(f"      已加载 {len(svc._ingestion.parent_recipes)} 篇已持久化的菜谱。")

    # ── 2. Parse all recipe .md files ─────────────────────────────────────
    print(f"[2/4] 扫描 {DISHES_DIR} 解析菜谱...")
    recipes = parse_all_recipes(DISHES_DIR)
    print(f"      解析到 {len(recipes)} 道菜谱。")

    # ── 3. Ingest only new recipes ────────────────────────────────────────
    existing = len(svc._ingestion.parent_recipes)
    new_recipes = [r for r in recipes if r.name not in
                   {p.recipe_name for p in svc._ingestion.parent_recipes.values()}]

    if not new_recipes:
        print("[3/4] 无新菜谱需要入库。")
    else:
        print(f"[3/4] 入库 {len(new_recipes)} 道新菜谱（共 {len(recipes)} 道）...")
        t0 = time.time()

        # Embedding API may have rate limits — ingest in batches of 5
        batch_size = 5
        total = len(new_recipes)
        for i in range(0, total, batch_size):
            batch = new_recipes[i : i + batch_size]
            try:
                svc.ingest_recipes(batch)
            except Exception as exc:
                logger.error("Batch ingestion failed at recipe %d: %s", i, exc)
            # Persist after each batch so partial progress isn't lost on crash
            svc.shutdown()
            elapsed = time.time() - t0
            done = min(i + batch_size, total)
            print(f"        进度 {done}/{total}  ({elapsed:.0f}s)")

        print(f"      入库完成！共 {len(svc._ingestion.parent_recipes)} 道菜谱。")

    # ── 4. Interactive query loop ─────────────────────────────────────────
    print()
    print("─" * 60)
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'stats' 查看当前菜谱统计")
    print("  输入 'rebuild' 强制重建 BM25 索引")
    print("─" * 60)

    try:
        while True:
            try:
                query = input("\n🔍 请输入你的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                print("再见！")
                break
            if query.lower() == "stats":
                parents = svc._ingestion.parent_recipes
                cuisine_counts: dict[str, int] = {}
                for p in parents.values():
                    raw = p.raw_data
                    ct = raw.get("cuisine_type") or "未知"
                    cuisine_counts[ct] = cuisine_counts.get(ct, 0) + 1
                print(f"\n📊 菜谱统计:")
                print(f"   总量: {len(parents)} 道")
                for ct, cnt in sorted(cuisine_counts.items()):
                    print(f"   {ct}: {cnt} 道")
                continue
            if query.lower() == "rebuild":
                svc._retriever.rebuild_bm25_index()
                print("BM25 索引已重建。")
                continue

            # ── Generate and stream answer ────────────────────────────────
            print("\n👨‍🍳 主厨回答: ", end="", flush=True)
            t0 = time.time()
            try:
                async for token in svc.process_query(query):
                    print(token, end="", flush=True)
            except Exception as exc:
                print(f"\n[!] 生成出错: {exc}")
            elapsed = time.time() - t0
            print(f"\n\n[{elapsed:.1f}s]")
            print("─" * 40)
    finally:
        svc.shutdown()
        print("菜谱数据已持久化。")


if __name__ == "__main__":
    asyncio.run(main())
