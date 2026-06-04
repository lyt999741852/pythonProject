"""RAG-specific configuration, extracted from ``app.config.Settings``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings


@dataclass
class RAGConfig:
    """RAG pipeline hyperparameters, sourced from pydantic Settings."""

    # ---- DashScope (embedding) ----
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    embedding_max_retries: int = 3
    embedding_retry_base_delay: float = 2.0

    # ---- Moonshot (LLM) ----
    moonshot_api_key: str = ""
    moonshot_base_url: str = "https://api.moonshot.cn/v1"
    llm_model: str = "moonshot-v1-8k"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ---- Chroma persistence ----
    chroma_persist_directory: str = "./data/chroma"

    # ---- Retrieval tuning ----
    vector_search_top_k: int = 5
    bm25_search_top_k: int = 5
    rrf_k_constant: int = 60
    final_top_n: int = 3

    # ---- Parent persistence ----
    parents_data_path: str = "./data/parents.json"

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGConfig:
        return cls(
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            dashscope_base_url=settings.DASHSCOPE_BASE_URL,
            embedding_model=settings.EMBEDDING_MODEL,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
            embedding_max_retries=settings.EMBEDDING_MAX_RETRIES,
            embedding_retry_base_delay=settings.EMBEDDING_RETRY_BASE_DELAY,
            moonshot_api_key=settings.MOONSHOT_API_KEY,
            moonshot_base_url=settings.MOONSHOT_BASE_URL,
            llm_model=settings.LLM_MODEL,
            llm_temperature=settings.LLM_TEMPERATURE,
            llm_max_tokens=settings.LLM_MAX_TOKENS,
            chroma_persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            vector_search_top_k=settings.VECTOR_SEARCH_TOP_K,
            bm25_search_top_k=settings.BM25_SEARCH_TOP_K,
            rrf_k_constant=settings.RRF_K_CONSTANT,
            final_top_n=settings.FINAL_TOP_N,
            parents_data_path=settings.PARENTS_DATA_PATH,
        )
