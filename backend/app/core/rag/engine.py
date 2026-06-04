"""Orchestration classes — the public API surface of the RAG pipeline.

Contains the 4 core classes that ``app.services.rag_service`` re-exports:

- ``RecipeIngestionService`` — ingest recipes into Chroma + parent dict
- ``RecipeRetriever`` — three-way recall (vector, parent-child, BM25) + RRF
- ``StreamingGenerator`` — async streaming LLM answer generation
- ``RAGService`` — top-level orchestrator
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import OrderedDict
from typing import AsyncGenerator, Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings
from app.core.rag.config import RAGConfig
from app.core.rag.data_processor import build_full_text, chunk_steps
from app.core.rag.embedder import create_embeddings
from app.core.rag.exceptions import (
    ChromaStorageError,
    EmbeddingError,
    RetrievalError,
)
from app.core.rag.keyword_search import _tokenize, build_bm25_index
from app.core.rag.llm import create_llm
from app.core.rag.prompt_manager import RAG_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, build_context
from app.core.rag.ranker import rrf_fuse
from app.core.rag.vector_store import create_vector_store
from app.models.recipe import ChildChunk, Recipe, RecipeParent, SearchResult, generate_id

logger = logging.getLogger(__name__)


# ===================================================================
# Module 1: Recipe Ingestion Service (Parent-Child Architecture)
# ===================================================================


class RecipeIngestionService:
    """Ingest recipes with a Parent-Child chunking strategy.

    **Parent**: full recipe text in an in-memory dict (persisted to JSON).
    **Child**: individual cooking steps embedded and stored in Chroma.
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        vector_store: Chroma,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._parent_recipes: dict[str, RecipeParent] = {}
        logger.info("RecipeIngestionService initialised with LangChain components.")

    # -- properties ---------------------------------------------------------

    @property
    def parent_recipes(self) -> dict[str, RecipeParent]:
        return self._parent_recipes

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store

    # -- embedding -----------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of texts via DashScope."""
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    # -- static helpers (backward-compatible wrappers around standalone funcs)

    @staticmethod
    def _build_full_text(recipe: Recipe) -> str:
        return build_full_text(recipe)

    @staticmethod
    def _chunk_steps(recipe: Recipe, parent_id: str) -> list[ChildChunk]:
        return chunk_steps(recipe, parent_id)

    # -- ingestion -----------------------------------------------------------

    def ingest_recipe(self, recipe: Recipe) -> str:
        """Ingest a single recipe.

        1. Build parent record and store in memory.
        2. Chunk steps into ChildChunks.
        3. Store child chunks as Documents in Chroma.
        """
        parent_id = generate_id()

        full_text = build_full_text(recipe)
        parent = RecipeParent(
            parent_id=parent_id,
            recipe_name=recipe.name,
            full_text=full_text,
            raw_data=recipe.model_dump(),
        )
        self._parent_recipes[parent_id] = parent
        logger.info("Parent recipe stored: '%s' (parent_id=%s)", recipe.name, parent_id)

        chunks = chunk_steps(recipe, parent_id)
        if not chunks:
            logger.warning("Recipe '%s' has no steps; no child chunks created.", recipe.name)
            return parent_id

        documents = [
            Document(page_content=c.text, metadata=c.metadata)
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]

        try:
            self._vector_store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            logger.error(
                "Chroma add failed for recipe '%s': %s — parent retained for BM25/keyword search.",
                recipe.name, exc,
            )
            # Parent recipe is kept even if vector storage fails,
            # so BM25/keyword search can still serve it.

        logger.info(
            "Ingested '%s' — %d chunks → Chroma (parent_id=%s)",
            recipe.name,
            len(chunks),
            parent_id,
        )
        return parent_id

    def ingest_recipes(self, recipes: list[Recipe]) -> list[str]:
        """Batch-ingest multiple recipes."""
        pids: list[str] = []
        for recipe in recipes:
            pid = self.ingest_recipe(recipe)
            pids.append(pid)
        return pids

    # -- parent persistence (JSON) ------------------------------------------

    def save_parents(self, filepath: str) -> None:
        """Serialize parent recipes to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {pid: p.model_dump() for pid, p in self._parent_recipes.items()}
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("Saved %d parent recipes to %s", len(data), filepath)

    def load_parents(self, filepath: str) -> None:
        """Restore parent recipes from JSON."""
        if not os.path.isfile(filepath):
            logger.info("No parent data file at %s — starting fresh", filepath)
            return
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for pid, raw in data.items():
            self._parent_recipes[pid] = RecipeParent(**raw)
        logger.info("Loaded %d parent recipes from %s", len(data), filepath)


# ===================================================================
# Module 2: Recipe Retriever (Three-Way Recall + RRF)
# ===================================================================


class RecipeRetriever:
    """Three-way recipe retrieval with Reciprocal Rank Fusion.

    Routes
    ------
    1. **Vector semantic** — Chroma ANN search on child chunks.
    2. **Parent-child关联** — extract parent_ids from route-1 → full parent text.
    3. **BM25 keyword** — keyword search on full parent texts.

    Route 2 enriches context only. Routes 1 & 3 are fused via RRF.
    """

    def __init__(
        self,
        vector_store: Chroma,
        ingestion_service: RecipeIngestionService,
        vector_search_top_k: int = 5,
        bm25_search_top_k: int = 5,
        rrf_k_constant: int = 60,
        final_top_n: int = 3,
    ) -> None:
        self._vector_store = vector_store
        self._ingestion = ingestion_service
        self._vector_search_top_k = vector_search_top_k
        self._bm25_search_top_k = bm25_search_top_k
        self._rrf_k = rrf_k_constant
        self._final_top_n = final_top_n

        self._bm25_retriever: Optional[BM25Retriever] = None

    # -- BM25 index management ----------------------------------------------

    def rebuild_bm25_index(self) -> None:
        """Build / rebuild the BM25 index from the current parent corpus."""
        self._bm25_retriever = build_bm25_index(
            self._ingestion.parent_recipes,
            self._bm25_search_top_k,
        )

    # -- retrieval routes ---------------------------------------------------

    def route1_vector_search(self, query: str) -> list[SearchResult]:
        """Route 1: vector semantic search on child chunks."""
        try:
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=self._vector_search_top_k,
                filter={"doc_type": "child"},
            )
        except Exception as exc:
            logger.error("Route-1 (Chroma query) failed: %s", exc)
            raise RetrievalError(f"Chroma query failed: {exc}") from exc

        search_results: list[SearchResult] = []
        for rank, (doc, distance) in enumerate(results, start=1):
            meta = doc.metadata
            score = 1.0 - distance if distance is not None else 0.0
            search_results.append(
                SearchResult(
                    parent_id=meta.get("parent_id", ""),
                    recipe_name=meta.get("recipe_name", ""),
                    text=doc.page_content,
                    score=score,
                    source="vector",
                    rank_vector=rank,
                )
            )
        return search_results

    def route2_parent_child_recall(
        self, vector_results: list[SearchResult]
    ) -> list[SearchResult]:
        """Route 2: enrich with full parent text (non-participating in RRF)."""
        parent_ids = list(OrderedDict.fromkeys(r.parent_id for r in vector_results if r.parent_id))
        parents = self._ingestion.parent_recipes

        results: list[SearchResult] = []
        for pid in parent_ids:
            parent = parents.get(pid)
            if parent is None:
                logger.warning("Route-2: parent_id=%s not found in store.", pid)
                continue
            results.append(
                SearchResult(
                    parent_id=pid,
                    recipe_name=parent.recipe_name,
                    text=parent.full_text,
                    score=0.0,
                    source="parent_child",
                )
            )
        return results

    def route3_bm25_search(self, query: str) -> list[SearchResult]:
        """Route 3: BM25 keyword search on full parent texts."""
        if self._bm25_retriever is None:
            logger.warning("Route-3: BM25 retriever not initialised; returning 0 results.")
            return []

        try:
            doc_results = self._bm25_retriever.invoke(query)
        except Exception as exc:
            logger.error("Route-3 (BM25 search) failed: %s", exc)
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

        parents = self._ingestion.parent_recipes
        results: list[SearchResult] = []
        for rank, doc in enumerate(doc_results, start=1):
            pid = doc.metadata.get("parent_id", "")
            parent = parents.get(pid)
            if parent is None:
                continue
            results.append(
                SearchResult(
                    parent_id=pid,
                    recipe_name=parent.recipe_name,
                    text=parent.full_text,
                    score=1.0,
                    source="bm25",
                    rank_bm25=rank,
                )
            )
        return results

    # -- RRF fusion ---------------------------------------------------------

    def rrf_fuse(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Fuse vector + BM25 results via RRF, then enrich with parent text."""
        fused_scores = rrf_fuse(
            vector_results,
            bm25_results,
            rrf_k=self._rrf_k,
            final_top_n=self._final_top_n,
        )

        parents = self._ingestion.parent_recipes
        fused: list[SearchResult] = []
        for pid, info in fused_scores.items():
            parent = parents.get(pid)
            text = parent.full_text if parent else ""
            fused.append(
                SearchResult(
                    parent_id=pid,
                    recipe_name="",
                    text=text,
                    score=info["score"],
                    source="vector+bm25",
                    rank_vector=info.get("rank_vector"),
                    rank_bm25=info.get("rank_bm25"),
                )
            )
        return fused

    # -- full pipeline ------------------------------------------------------

    def retrieve(
        self, query: str, query_embedding: list[float]
    ) -> list[SearchResult]:
        """Run the full 3-way retrieval pipeline with RRF fusion."""
        route1 = self.route1_vector_search(query)
        _ = self.route2_parent_child_recall(route1)  # enrichment only
        route3 = self.route3_bm25_search(query)
        fused = self.rrf_fuse(route1, route3)

        logger.info(
            "Retrieval: %d route-1 hits, %d route-3 hits → %d fused results",
            len(route1),
            len(route3),
            len(fused),
        )
        return fused


# ===================================================================
# Module 3: Streaming Generator (Kimi LLM)
# ===================================================================


class StreamingGenerator:
    """Stream answer generation via Kimi (Moonshot) LLM.

    Uses dual-mode prompts:
    - RAG mode: when retrieval context is available.
    - Chat mode: free conversation without recipe context.
    """

    def __init__(
        self,
        retriever: RecipeRetriever,
        ingestion_service: RecipeIngestionService,
        llm: ChatOpenAI,
    ) -> None:
        self._retriever = retriever
        self._ingestion = ingestion_service
        self._llm = llm

    # -- context building ---------------------------------------------------

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        return build_context(results)

    # -- streaming ----------------------------------------------------------

    async def generate_answer_stream(
        self,
        query: str,
        extra_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """Async generator that streams the LLM answer token by token.

        Args:
            query: The user's question.
            extra_context: Optional prefix context (e.g. health calculation
                results) injected before the retrieval context.
        """
        # --- Step 1: embed query ---
        results: list[SearchResult] = []
        query_embedding: list[float] | None = None

        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, self._ingestion.embed_texts, [query]
            )
            if embeddings:
                query_embedding = embeddings[0]
        except EmbeddingError:
            logger.warning("Query embedding failed (DashScope quota?) — skipping RAG retrieval")

        # --- Step 2: retrieve (only if embedding succeeded) ---
        if query_embedding:
            try:
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(
                    None, self._retriever.retrieve, query, query_embedding
                )
            except RetrievalError:
                logger.warning("Retrieval failed — falling back to chat mode")

        # --- Step 3: choose prompt mode based on whether results exist ---
        context = ""
        if results:
            context = self._build_context(results)
            if extra_context:
                context = f"{extra_context}\n\n{context}"
            system_prompt = RAG_SYSTEM_PROMPT
            human_template = (
                "## 用户问题\n{query}\n\n"
                "## 参考上下文\n{context}\n\n"
                "请根据以上参考上下文回答用户的问题。"
            )
        else:
            # No recipe results → chat mode
            system_prompt = CHAT_SYSTEM_PROMPT
            human_template = "{query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template),
        ])
        kwargs = {"query": query}
        if results:
            kwargs["context"] = context
            if extra_context:
                kwargs["extra_context"] = extra_context
        messages = prompt.format_messages(**kwargs)

        # --- Step 4: stream from Moonshot ---
        try:
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception:
            logger.exception("Moonshot stream failed")
            yield "\n\n[主厨提示：回答因网络波动中断，请重试。]"
            return


# ===================================================================
# Module 4: Top-Level Orchestrator (RAGService)
# ===================================================================


class RAGService:
    """Top-level orchestrator tying ingestion, retrieval, and generation.

    This is the public API surface for the rest of the application.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        config = RAGConfig.from_settings(self._settings)

        # --- LangChain components ---
        self._embeddings = create_embeddings(config)
        self._vector_store = create_vector_store(config, self._embeddings)
        self._llm = create_llm(config)

        self._ingestion = RecipeIngestionService(
            embeddings=self._embeddings,
            vector_store=self._vector_store,
        )

        self._retriever = RecipeRetriever(
            vector_store=self._vector_store,
            ingestion_service=self._ingestion,
            vector_search_top_k=config.vector_search_top_k,
            bm25_search_top_k=config.bm25_search_top_k,
            rrf_k_constant=config.rrf_k_constant,
            final_top_n=config.final_top_n,
        )

        self._generator = StreamingGenerator(
            retriever=self._retriever,
            ingestion_service=self._ingestion,
            llm=self._llm,
        )

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> None:
        """Load persisted parents and rebuild BM25 index."""
        parents_path = self._settings.PARENTS_DATA_PATH
        self._ingestion.load_parents(parents_path)
        self._retriever.rebuild_bm25_index()
        logger.info(
            "RAGService initialised with %d parent recipes.",
            len(self._ingestion.parent_recipes),
        )

    def shutdown(self) -> None:
        """Persist parent data before shutting down."""
        parents_path = self._settings.PARENTS_DATA_PATH
        self._ingestion.save_parents(parents_path)
        logger.info("RAGService shut down — parent data persisted.")

    # -- ingestion API ------------------------------------------------------

    def ingest_recipe(self, recipe: Recipe) -> str:
        """Ingest one recipe and refresh the BM25 index."""
        pid = self._ingestion.ingest_recipe(recipe)
        self._retriever.rebuild_bm25_index()
        return pid

    def ingest_recipes(self, recipes: list[Recipe]) -> list[str]:
        """Ingest multiple recipes and refresh the BM25 index once."""
        pids = self._ingestion.ingest_recipes(recipes)
        self._retriever.rebuild_bm25_index()
        return pids

    # -- retrieval API ------------------------------------------------------

    def retrieve(self, query: str) -> list[SearchResult]:
        """Synchronous retrieval for callers who only need raw results."""
        try:
            embeddings = self._ingestion.embed_texts([query])
        except EmbeddingError:
            return []
        if not embeddings:
            return []
        return self._retriever.retrieve(query, embeddings[0])

    # -- streaming generation API -------------------------------------------

    async def generate_answer_stream(
        self, query: str
    ) -> AsyncGenerator[str, None]:
        """Async generator — stream the answer for a user query."""
        async for token in self._generator.generate_answer_stream(query):
            yield token

    # -- unified query API (with routing) -----------------------------------

    async def process_query(
        self, query: str
    ) -> AsyncGenerator[str, None]:
        """Unified entry-point for all user queries.

        Automatically routes between recipe-RAG and free chat.
        """
        async for token in self.generate_answer_stream(query):
            yield token
