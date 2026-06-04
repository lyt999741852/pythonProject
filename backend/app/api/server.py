"""
FastAPI server with SSE streaming for Vibe Cooking.

Endpoints
---------
POST /api/chat    — SSE streaming chat (accepts {"query": "..."})
GET  /api/stats   — recipe count by cuisine type
GET  /api/health  — health check
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Ensure the backend directory is on sys.path
_backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.config import Settings
from app.recipe_parser import parse_all_recipes
from app.services.rag_service import RAGService

logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Global RAG service instance (initialised during lifespan)
# ---------------------------------------------------------------------------

rag: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init RAG service + ingest recipes. Shutdown: persist data."""
    global rag
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting Vibe Cooking API server ...")
    settings = Settings()
    rag = RAGService(settings=settings)
    rag.initialize()
    logger.info("Loaded %d persisted parent recipes.", len(rag._ingestion.parent_recipes))

    # ── Scan dishes/ and ingest new recipes ──
    dishes_dir = os.path.join(_backend_dir, "dishes")
    if os.path.isdir(dishes_dir):
        recipes = parse_all_recipes(dishes_dir)
        existing_names = {p.recipe_name for p in rag._ingestion.parent_recipes.values()}
        new_recipes = [r for r in recipes if r.name not in existing_names]
        if new_recipes:
            logger.info("Ingesting %d new recipes ...", len(new_recipes))
            t0 = time.time()
            for i in range(0, len(new_recipes), 5):
                batch = new_recipes[i:i + 5]
                try:
                    rag.ingest_recipes(batch)
                except Exception as exc:
                    logger.error("Batch ingestion failed at %d: %s", i, exc)
                rag._ingestion.save_parents(rag._settings.PARENTS_DATA_PATH)  # direct persist, no lifecycle trigger
            logger.info("Ingested %d recipes in %.0fs.", len(new_recipes), time.time() - t0)
        else:
            logger.info("No new recipes to ingest.")
    else:
        logger.warning("Dishes directory not found at %s", dishes_dir)

    yield
    rag.shutdown()
    logger.info("Shutdown complete — parent data persisted.")


app = FastAPI(title="Vibe Cooking API", version="0.1.0", lifespan=lifespan)

# Allow frontend dev server (localhost:5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str


class StatsResponse(BaseModel):
    total: int
    by_cuisine: dict[str, int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/routes")
async def list_routes() -> list[dict]:
    """Debug endpoint: list all registered routes."""
    return [
        {"path": route.path, "method": list(route.methods), "name": route.name}
        for route in app.routes
        if isinstance(route, APIRoute)
    ]


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/stats")
async def stats() -> StatsResponse:
    if rag is None:
        return StatsResponse(total=0, by_cuisine={})
    parents = rag._ingestion.parent_recipes
    counts: dict[str, int] = {}
    for p in parents.values():
        ct = p.raw_data.get("cuisine_type") or "未知"
        counts[ct] = counts.get(ct, 0) + 1
    return StatsResponse(total=len(parents), by_cuisine=counts)


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """SSE streaming chat endpoint."""

    async def event_stream() -> AsyncGenerator[str, None]:
        if rag is None:
            yield f"data: {json.dumps({'error': 'Service not initialised'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # SSE initial comment — flushes response headers immediately
        yield ": connected\n\n"

        try:
            async for token in rag.process_query(body.query):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.exception("Chat error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8001, reload=False)
