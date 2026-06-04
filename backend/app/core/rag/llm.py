"""LangChain ChatOpenAI factory (Moonshot / Kimi-compatible).

Encapsulates Moonshot-specific endpoint configuration.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.rag.config import RAGConfig


def create_llm(config: RAGConfig) -> ChatOpenAI:
    """Create a ``ChatOpenAI`` instance configured for Moonshot (Kimi).

    Args:
        config: RAG pipeline configuration.

    Returns:
        A fully initialised ``ChatOpenAI`` that streams from Moonshot.
    """
    return ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.moonshot_api_key,
        openai_api_base=config.moonshot_base_url,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        streaming=True,
    )
