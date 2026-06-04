"""Custom exceptions for the RAG pipeline."""


class RAGServiceError(Exception):
    """Base exception for all RAG service errors."""


class EmbeddingError(RAGServiceError):
    """DashScope embedding API failures (auth, timeout, rate-limit)."""


class RetrievalError(RAGServiceError):
    """Retrieval pipeline failures (Chroma, BM25)."""


class GenerationError(RAGServiceError):
    """Moonshot LLM streaming failures."""


class ChromaStorageError(RAGServiceError):
    """Chroma persistence / connection failures."""
