"""Tests for StreamingGenerator and RAGService."""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from app.models.recipe import Recipe, RecipeParent, SearchResult
from app.services.rag_service import (
    EmbeddingError,
    RAGService,
    RecipeIngestionService,
    RecipeRetriever,
    StreamingGenerator,
)
from app.core.rag.prompt_manager import RAG_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT


# -- helpers -----------------------------------------------------------------


def create_mock_chunk(content: str) -> MagicMock:
    """Create a mock AIMessageChunk with given content."""
    chunk = MagicMock()
    chunk.content = content
    return chunk


async def async_generator_from_list(items: list) -> AsyncGenerator:
    """Convert a list into an async generator for mocking."""
    for item in items:
        yield item


# -- StreamingGenerator ------------------------------------------------------


@pytest.fixture
def mock_retriever() -> MagicMock:
    """A mocked RecipeRetriever that returns known results."""
    ret = MagicMock(spec=RecipeRetriever)
    ret.retrieve.return_value = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="宫保鸡丁做法...", score=0.9, source="vector+bm25"),
    ]
    return ret


@pytest.fixture
def mock_ingestion() -> MagicMock:
    """A mocked RecipeIngestionService with embed_texts."""
    ing = MagicMock(spec=RecipeIngestionService)
    ing.embed_texts.return_value = [[0.1] * 4]
    ing.parent_recipes = {
        "p1": RecipeParent(parent_id="p1", recipe_name="宫保鸡丁", full_text="宫保鸡丁做法...", raw_data={}),
    }
    return ing


@pytest.fixture
def generator(mock_retriever: MagicMock, mock_ingestion: MagicMock) -> StreamingGenerator:
    """StreamingGenerator backed by mocked dependencies."""
    llm = MagicMock()
    return StreamingGenerator(
        retriever=mock_retriever,
        ingestion_service=mock_ingestion,
        llm=llm,
    )


@pytest.mark.asyncio
async def test_generate_answer_stream_yields_tokens(generator: StreamingGenerator):
    """StreamingGenerator should yield tokens from the LLM stream."""
    generator._llm.astream = MagicMock(
        return_value=async_generator_from_list(
            [create_mock_chunk("宫保"), create_mock_chunk("鸡丁"), create_mock_chunk("做法")]
        )
    )

    collected = []
    async for token in generator.generate_answer_stream("宫保鸡丁怎么做？"):
        collected.append(token)

    assert len(collected) == 3
    assert collected[0] == "宫保"
    assert collected[1] == "鸡丁"
    assert collected[2] == "做法"


@pytest.mark.asyncio
async def test_generate_answer_stream_fallback_no_results(generator: StreamingGenerator):
    """When retrieval returns empty, generator should yield LLM chat response."""
    generator._retriever.retrieve.return_value = []
    generator._llm.astream = MagicMock(
        return_value=async_generator_from_list(
            [create_mock_chunk("你好，有什么可以帮你的吗？")]
        )
    )

    collected = []
    async for token in generator.generate_answer_stream("不存在的菜"):
        collected.append(token)

    assert len(collected) == 1
    assert "你好" in collected[0]


@pytest.mark.asyncio
async def test_generate_answer_stream_embedding_failure(generator: StreamingGenerator):
    """When embedding fails, generator should yield a friendly error message."""
    generator._ingestion.embed_texts.side_effect = EmbeddingError("API timeout")

    collected = []
    async for token in generator.generate_answer_stream("test"):
        collected.append(token)

    assert len(collected) == 1
    assert "技术问题" in collected[0]


@pytest.mark.asyncio
async def test_generate_answer_stream_moonshot_error(generator: StreamingGenerator):
    """When Moonshot API call fails, generator should yield a friendly error."""
    generator._llm.astream = MagicMock(side_effect=Exception("Moonshot down"))

    collected = []
    async for token in generator.generate_answer_stream("test"):
        collected.append(token)

    assert len(collected) == 1
    assert "网络波动" in collected[0] or "技术问题" in collected[0]


@pytest.mark.asyncio
async def test_generate_answer_stream_stream_interrupted(generator: StreamingGenerator):
    """When stream breaks mid-way, generator should yield interruption message."""
    async def _broken_stream():
        yield create_mock_chunk("token1")
        yield create_mock_chunk("token2")
        raise Exception("Network disconnected")

    generator._llm.astream = MagicMock(return_value=_broken_stream())

    collected = []
    async for token in generator.generate_answer_stream("test"):
        collected.append(token)

    # Should have gotten the 2 real tokens + the interruption message
    assert len(collected) >= 2
    assert "网络波动" in collected[-1]


def test_build_context(mock_retriever: MagicMock, mock_ingestion: MagicMock):
    """_build_context should format results as numbered references."""
    gen = StreamingGenerator(
        retriever=mock_retriever,
        ingestion_service=mock_ingestion,
        llm=MagicMock(),
    )
    results = [
        SearchResult(parent_id="p1", recipe_name="宫保鸡丁", text="宫保鸡丁做法...", score=0.9, source="vector+bm25"),
    ]
    context = gen._build_context(results)
    assert "[参考 1: 宫保鸡丁]" in context
    assert "宫保鸡丁做法..." in context


def test_rag_system_prompt_content():
    """RAG_SYSTEM_PROMPT should contain the core rules."""
    from app.core.rag.prompt_manager import RAG_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
    assert "Vibe Cooking" in RAG_SYSTEM_PROMPT
    assert "灵活" in RAG_SYSTEM_PROMPT or "格式灵活" in RAG_SYSTEM_PROMPT
    assert "不要编造" in RAG_SYSTEM_PROMPT
    assert "CHAT_SYSTEM_PROMPT" in dir() or CHAT_SYSTEM_PROMPT


def test_chat_system_prompt_content():
    """CHAT_SYSTEM_PROMPT should support free chat."""
    from app.core.rag.prompt_manager import CHAT_SYSTEM_PROMPT
    assert "自由对话" in CHAT_SYSTEM_PROMPT
    assert "自由聊天" in CHAT_SYSTEM_PROMPT or "自由" in CHAT_SYSTEM_PROMPT


# -- RAGService orchestrator -------------------------------------------------


@pytest.mark.asyncio
async def test_rag_service_generate_stream(tmp_path):
    """RAGService.generate_answer_stream should work end-to-end with mocks."""
    settings = MagicMock()
    settings.CHROMA_PERSIST_DIRECTORY = str(tmp_path / "chroma")
    settings.DASHSCOPE_API_KEY = "test-ds"
    settings.DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    settings.EMBEDDING_MODEL = "text-embedding-v3"
    settings.EMBEDDING_DIMENSIONS = 4
    settings.EMBEDDING_MAX_RETRIES = 1
    settings.EMBEDDING_RETRY_BASE_DELAY = 1.0
    settings.MOONSHOT_API_KEY = "test-ms"
    settings.MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
    settings.LLM_MODEL = "moonshot-v1-8k"
    settings.LLM_TEMPERATURE = 0.3
    settings.LLM_MAX_TOKENS = 128
    settings.VECTOR_SEARCH_TOP_K = 3
    settings.BM25_SEARCH_TOP_K = 3
    settings.RRF_K_CONSTANT = 60
    settings.FINAL_TOP_N = 2
    settings.PARENTS_DATA_PATH = str(tmp_path / "parents.json")

    with patch("app.core.rag.embedder.OpenAIEmbeddings", return_value=MagicMock()), \
         patch("app.core.rag.vector_store.Chroma", return_value=MagicMock()), \
         patch("app.core.rag.llm.ChatOpenAI", return_value=MagicMock()):
        svc = RAGService(settings=settings)

    # Mock ingestion.embed_texts
    svc._ingestion.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])

    # Mock Chroma query (no real data)
    svc._retriever._vector_store.similarity_search_with_score = MagicMock(return_value=[])

    # Mock Moonshot stream
    svc._generator._llm.astream = MagicMock(
        return_value=async_generator_from_list(
            [create_mock_chunk("你好"), create_mock_chunk("世界")]
        )
    )

    collected = []
    async for token in svc.generate_answer_stream("test"):
        collected.append(token)

    # With no results, should use chat mode → stream mock tokens
    assert len(collected) >= 1
    assert "你好" in collected[0] or "世界" in "".join(collected)
