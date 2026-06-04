"""LangChain OpenAIEmbeddings factory (DashScope-compatible).

Kept as a dedicated module to encapsulate DashScope-specific configuration
(``chunk_size=10``, ``check_embedding_ctx_length=False``).
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.core.rag.config import RAGConfig


def create_embeddings(config: RAGConfig) -> OpenAIEmbeddings:
    """Create an ``OpenAIEmbeddings`` instance configured for DashScope.

    Args:
        config: RAG pipeline configuration.

    Returns:
        A fully initialised ``OpenAIEmbeddings`` that talks to
        DashScope's OpenAI-compatible endpoint.
    """
    return OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.dashscope_api_key,
        openai_api_base=config.dashscope_base_url,
        dimensions=config.embedding_dimensions,
        max_retries=config.embedding_max_retries,
        chunk_size=10,
        check_embedding_ctx_length=False,
    )
