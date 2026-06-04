"""Tests for RecipeRetriever (three-way recall + RRF fusion)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.models.recipe import Recipe, RecipeParent, SearchResult
from app.services.rag_service import (
    RecipeIngestionService,
    RecipeRetriever,
    RetrievalError,
    ChromaStorageError,
    _tokenize,
)


# -- tokenizer ---------------------------------------------------------------


def test_tokenize_chinese():
    """Chinese text should be segmented via jieba."""
    tokens = _tokenize("宫保鸡丁怎么做")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # jieba should split "宫保鸡丁" into at least 2 tokens
    assert any(len(t) <= 2 for t in tokens)  # some short tokens exist


def test_tokenize_mixed():
    """Mixed Chinese-English text should be handled correctly."""
    tokens = _tokenize("巴沙鱼 fillet 做法")
    assert "巴沙鱼" in tokens or "巴" in tokens or "沙" in tokens or "鱼" in tokens
    assert "fillet" in tokens


def test_tokenize_english():
    """English-only text should be lowercased and split."""
    tokens = _tokenize("How to cook fish")
    assert "how" in tokens
    assert "cook" in tokens
    assert "fish" in tokens


def test_tokenize_empty():
    """Empty string should produce an empty token list."""
    tokens = _tokenize("")
    assert tokens == []


# -- fixture helpers ---------------------------------------------------------


@pytest.fixture
def ingestion_with_parents() -> RecipeIngestionService:
    """Return an ingestion service with 3 pre-loaded parent recipes but no real Chroma data."""
    embeddings = MagicMock(spec=OpenAIEmbeddings)
    vector_store = MagicMock(spec=Chroma)
    service = RecipeIngestionService(embeddings=embeddings, vector_store=vector_store)
    # Manually inject parent recipes (bypass embedding)
    service._parent_recipes = {
        "p1": RecipeParent(parent_id="p1", recipe_name="宫保鸡丁", full_text="宫保鸡丁 鸡肉 花生 辣椒 川菜 辣", raw_data={}),
        "p2": RecipeParent(parent_id="p2", recipe_name="西红柿鸡蛋汤", full_text="西红柿鸡蛋汤 西红柿 鸡蛋 清淡 汤", raw_data={}),
        "p3": RecipeParent(parent_id="p3", recipe_name="巴沙鱼柳", full_text="巴沙鱼柳 鱼肉 柠檬 黑胡椒 西餐", raw_data={}),
    }
    return service


@pytest.fixture
def retriever(ingestion_with_parents: RecipeIngestionService) -> RecipeRetriever:
    """Retriever backed by the fixture above, with BM25 index."""
    r = RecipeRetriever(
        vector_store=ingestion_with_parents.vector_store,
        ingestion_service=ingestion_with_parents,
        vector_search_top_k=3,
        bm25_search_top_k=3,
        rrf_k_constant=60,
        final_top_n=2,
    )
    r.rebuild_bm25_index()
    return r


# -- BM25 index management ---------------------------------------------------


def test_rebuild_bm25_index(retriever: RecipeRetriever):
    """After rebuild, the BM25 retriever should be initialised."""
    assert retriever._bm25_retriever is not None


def test_rebuild_bm25_empty(ingestion_with_parents: RecipeIngestionService):
    """Empty parent corpus should produce None retriever."""
    ingestion_with_parents._parent_recipes = {}
    r = RecipeRetriever(
        vector_store=ingestion_with_parents.vector_store,
        ingestion_service=ingestion_with_parents,
    )
    r.rebuild_bm25_index()
    assert r._bm25_retriever is None


# -- Route 1: vector search --------------------------------------------------


def test_route1_vector_search_happy(retriever: RecipeRetriever):
    """Route 1 should return results ranked by Chroma distances."""
    mock_docs = [
        (Document(page_content="chicken text", metadata={"parent_id": "p1", "recipe_name": "宫保鸡丁", "doc_type": "child"}), 0.1),
        (Document(page_content="tomato text", metadata={"parent_id": "p2", "recipe_name": "西红柿鸡蛋汤", "doc_type": "child"}), 0.3),
    ]
    retriever._vector_store.similarity_search_with_score = MagicMock(return_value=mock_docs)
    results = retriever.route1_vector_search("鸡肉怎么做")
    assert len(results) == 2
    assert results[0].parent_id == "p1"
    assert results[0].rank_vector == 1
    assert results[1].parent_id == "p2"
    assert results[1].rank_vector == 2


def test_route1_vector_search_empty(retriever: RecipeRetriever):
    """Route 1 should return empty list when Chroma returns no hits."""
    retriever._vector_store.similarity_search_with_score = MagicMock(return_value=[])
    results = retriever.route1_vector_search("鸡肉怎么做")
    assert results == []


def test_route1_vector_search_chroma_error(retriever: RecipeRetriever):
    """Route 1 should raise RetrievalError on Chroma failure."""
    retriever._vector_store.similarity_search_with_score = MagicMock(
        side_effect=Exception("Chroma crashed")
    )
    with pytest.raises(RetrievalError):
        retriever.route1_vector_search("鸡肉怎么做")


# -- Route 2: parent-child recall --------------------------------------------


def test_route2_parent_child_recall(retriever: RecipeRetriever):
    """Route 2 should return full parent text for parent_ids from route-1."""
    vector_results = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="chunk", score=0.9, source="vector"),
        SearchResult(parent_id="p3", recipe_name="巴沙鱼柳", text="chunk", score=0.8, source="vector"),
    ]
    results = retriever.route2_parent_child_recall(vector_results)
    assert len(results) == 2
    assert results[0].parent_id == "p1"
    assert "宫保鸡丁" in results[0].text
    assert results[1].parent_id == "p3"
    assert "巴沙鱼柳" in results[1].text


def test_route2_parent_child_recall_dedup(retriever: RecipeRetriever):
    """Route 2 should deduplicate parent_ids."""
    vector_results = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="chunk1", score=0.9, source="vector"),
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="chunk2", score=0.7, source="vector"),
    ]
    results = retriever.route2_parent_child_recall(vector_results)
    assert len(results) == 1


def test_route2_missing_parent(retriever: RecipeRetriever):
    """Route 2 should silently skip parent_ids not in the store."""
    vector_results = [
        SearchResult(parent_id="nonexistent", recipe_name="?", text="chunk", score=0.5, source="vector"),
    ]
    results = retriever.route2_parent_child_recall(vector_results)
    assert results == []


# -- Route 3: BM25 search ----------------------------------------------------


def test_route3_bm25_search(retriever: RecipeRetriever):
    """Route 3 should return parent-level results ranked by BM25 score."""
    results = retriever.route3_bm25_search("鸡肉 辣椒 川菜")
    assert len(results) > 0
    # The most relevant document should be 宫保鸡丁
    top = results[0]
    assert top.parent_id == "p1"
    assert top.source == "bm25"
    assert top.rank_bm25 == 1


def test_route3_bm25_search_empty_index(ingestion_with_parents: RecipeIngestionService):
    """Route 3 should return empty list when BM25 retriever is None."""
    r = RecipeRetriever(
        vector_store=ingestion_with_parents.vector_store,
        ingestion_service=ingestion_with_parents,
    )
    results = r.route3_bm25_search("anything")
    assert results == []


# -- RRF fusion --------------------------------------------------------------


def test_rrf_fuse_both_lists(retriever: RecipeRetriever):
    """RRF fusion should boost parents appearing in both lists."""
    vec = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="", score=0.9, source="vector", rank_vector=1),
        SearchResult(parent_id="p2", recipe_name="西红柿鸡蛋汤", text="", score=0.7, source="vector", rank_vector=2),
    ]
    bm = [
        SearchResult(parent_id="p2", recipe_name="西红柿鸡蛋汤", text="", score=10.0, source="bm25", rank_bm25=1),
        SearchResult(parent_id="p3", recipe_name="巴沙鱼柳", text="", score=5.0, source="bm25", rank_bm25=2),
    ]
    fused = retriever.rrf_fuse(vec, bm)
    assert len(fused) == 2  # final_top_n = 2
    # p2 appears in both lists -> should be ranked first
    assert fused[0].parent_id == "p2"
    assert fused[0].source == "vector+bm25"


def test_rrf_fuse_only_vector(retriever: RecipeRetriever):
    """RRF with only vector results should still rank correctly."""
    vec = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="", score=0.9, source="vector", rank_vector=1),
        SearchResult(parent_id="p2", recipe_name="西红柿鸡蛋汤", text="", score=0.7, source="vector", rank_vector=2),
    ]
    fused = retriever.rrf_fuse(vec, [])
    assert len(fused) == 2
    assert fused[0].parent_id == "p1"


def test_rrf_fuse_only_bm25(retriever: RecipeRetriever):
    """RRF with only BM25 results should still rank correctly."""
    bm = [
        SearchResult(parent_id="p2", recipe_name="西红柿鸡蛋汤", text="", score=10.0, source="bm25", rank_bm25=1),
        SearchResult(parent_id="p3", recipe_name="巴沙鱼柳", text="", score=5.0, source="bm25", rank_bm25=2),
    ]
    fused = retriever.rrf_fuse([], bm)
    assert len(fused) == 2
    assert fused[0].parent_id == "p2"


def test_rrf_fuse_empty(retriever: RecipeRetriever):
    """RRF with no results should return empty list."""
    assert retriever.rrf_fuse([], []) == []


def test_rrf_fuse_rank_values(retriever: RecipeRetriever):
    """RRF scores should be positive and follow the 1/(k+rank) formula."""
    vec = [
        SearchResult(parent_id="p1", recipe_name="A", text="", score=0.9, source="vector", rank_vector=1),
    ]
    bm = [
        SearchResult(parent_id="p1", recipe_name="A", text="", score=5.0, source="bm25", rank_bm25=1),
    ]
    fused = retriever.rrf_fuse(vec, bm)
    expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert abs(fused[0].score - expected_score) < 1e-6


# -- Full pipeline -----------------------------------------------------------


def test_retrieve_full_pipeline(retriever: RecipeRetriever):
    """The full retrieve() pipeline should return fused results."""
    mock_docs = [
        (Document(page_content="chicken text", metadata={"parent_id": "p1", "recipe_name": "宫保鸡丁", "doc_type": "child"}), 0.15),
    ]
    retriever._vector_store.similarity_search_with_score = MagicMock(return_value=mock_docs)
    results = retriever.retrieve("鸡肉怎么做", [0.1, 0.2, 0.3, 0.4])
    assert len(results) > 0
    assert all(r.source == "vector+bm25" for r in results)


def test_retrieve_no_parents(ingestion_with_parents: RecipeIngestionService):
    """Retrieval on empty store should return empty list."""
    ingestion_with_parents._parent_recipes = {}
    r = RecipeRetriever(
        vector_store=ingestion_with_parents.vector_store,
        ingestion_service=ingestion_with_parents,
    )
    r.rebuild_bm25_index()
    r._vector_store.similarity_search_with_score = MagicMock(return_value=[])
    results = r.retrieve("anything", [0.1, 0.2, 0.3, 0.4])
    assert results == []
