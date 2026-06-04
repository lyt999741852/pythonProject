"""Chinese-aware BM25 keyword search via LangChain ``BM25Retriever``.

Provides the shared ``_tokenize`` function (jieba-based) and a factory
``build_bm25_index`` that constructs a ``BM25Retriever`` from LangChain.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.models.recipe import RecipeParent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chinese / English tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese-English text for BM25 indexing.

    - Chinese sequences are segmented via ``jieba.lcut``.
    - Non-Chinese tokens are lowercased and split on whitespace.
    """
    tokens: list[str] = []
    parts = re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", text.lower())
    for part in parts:
        if re.match(r"^[一-鿿]+$", part):
            tokens.extend(jieba.lcut(part))
        else:
            tokens.append(part)
    return tokens


# ---------------------------------------------------------------------------
# BM25 index factory
# ---------------------------------------------------------------------------


def build_bm25_index(
    parents: dict[str, RecipeParent],
    top_k: int,
) -> Optional[BM25Retriever]:
    """Build a ``BM25Retriever`` from the current parent corpus.

    Args:
        parents: Map of ``parent_id -> RecipeParent``.
        top_k: Number of documents to return per query.

    Returns:
        A ``BM25Retriever`` instance, or ``None`` if the corpus is empty.
    """
    if not parents:
        logger.info("BM25 index: empty corpus, index cleared.")
        return None

    corpus = [
        Document(page_content=p.full_text, metadata={"parent_id": pid})
        for pid, p in parents.items()
    ]

    retriever = BM25Retriever.from_documents(
        corpus,
        k=top_k,
        preprocess_func=_tokenize,
    )
    logger.info("BM25 index rebuilt with %d documents.", len(corpus))
    return retriever
