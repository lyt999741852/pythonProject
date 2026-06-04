"""LangChain Chroma vector-store factory (langchain_chroma).

Encapsulates Chroma collection initialisation with cosine distance.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.rag.config import RAGConfig


def create_vector_store(config: RAGConfig, embeddings: OpenAIEmbeddings) -> Chroma:
    """Create a ``Chroma`` vector store backed by a local persistent directory.

    Args:
        config: RAG pipeline configuration.
        embeddings: The ``OpenAIEmbeddings`` instance used for
            automatic query embedding.

    Returns:
        A persistent ``Chroma`` collection with cosine-distance HNSW index.
    """
    return Chroma(
        collection_name="recipe_chunks",
        embedding_function=embeddings,
        persist_directory=config.chroma_persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
