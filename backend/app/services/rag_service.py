"""
Vibe Cooking — RAG Service Facade

Backward-compatible re-exports from ``app.core.rag``.
All RAG logic now lives in the modular ``app/core/rag/`` package.
"""

from app.core.rag import (
    ChromaStorageError,
    EmbeddingError,
    GenerationError,
    RAGServiceError,
    RetrievalError,
    RAGService,
    RecipeIngestionService,
    RecipeRetriever,
    StreamingGenerator,
    _tokenize,
)

__all__ = [
    "ChromaStorageError",
    "EmbeddingError",
    "GenerationError",
    "RAGServiceError",
    "RetrievalError",
    "RAGService",
    "RecipeIngestionService",
    "RecipeRetriever",
    "StreamingGenerator",
    "_tokenize",
]
