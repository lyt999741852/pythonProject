"""Tests for RecipeIngestionService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.models.recipe import Recipe, generate_id
from app.services.rag_service import (
    ChromaStorageError,
    EmbeddingError,
    RecipeIngestionService,
)


@pytest.fixture
def ingestion_service() -> RecipeIngestionService:
    """Create a RecipeIngestionService with mocked dependencies."""
    embeddings = MagicMock(spec=OpenAIEmbeddings)
    vector_store = MagicMock(spec=Chroma)
    return RecipeIngestionService(embeddings=embeddings, vector_store=vector_store)


# -- embedding ---------------------------------------------------------------


def test_ingest_single_recipe(ingestion_service: RecipeIngestionService, sample_recipe: Recipe):
    """Ingesting one recipe should create one parent and store N child Documents in Chroma."""
    pid = ingestion_service.ingest_recipe(sample_recipe)

    # Parent stored in memory
    assert pid in ingestion_service.parent_recipes
    parent = ingestion_service.parent_recipes[pid]
    assert parent.recipe_name == "宫保鸡丁"
    assert "鸡胸肉" in parent.full_text
    assert "火候要快" in parent.full_text

    # Chroma add_documents was called with correct number of docs
    ingestion_service._vector_store.add_documents.assert_called_once()
    call_kwargs = ingestion_service._vector_store.add_documents.call_args[1]
    docs = call_kwargs["documents"]
    assert len(docs) == 3  # 3 steps
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.metadata.get("doc_type") == "child" for d in docs)


def test_ingest_multiple_recipes(ingestion_service: RecipeIngestionService, sample_recipes: list[Recipe]):
    """Batch ingestion produces correct number of parents."""
    pids = ingestion_service.ingest_recipes(sample_recipes)
    assert len(pids) == 3
    assert len(ingestion_service.parent_recipes) == 3


def test_ingest_recipe_no_steps(ingestion_service: RecipeIngestionService):
    """A recipe with no steps should still be stored as a parent (no child chunks)."""
    recipe = Recipe(name="空菜谱", steps=[])
    pid = ingestion_service.ingest_recipe(recipe)
    assert pid in ingestion_service.parent_recipes
    ingestion_service._vector_store.add_documents.assert_not_called()


def test_ingest_and_count_parents(ingestion_service: RecipeIngestionService, sample_recipe: Recipe):
    """Parent count matches ingested count."""
    assert len(ingestion_service.parent_recipes) == 0
    ingestion_service.ingest_recipe(sample_recipe)
    assert len(ingestion_service.parent_recipes) == 1


# -- full text builder -------------------------------------------------------


def test_build_full_text_contains_all_sections(sample_recipe: Recipe):
    """_build_full_text should include recipe name, ingredients, steps, tips."""
    text = RecipeIngestionService._build_full_text(sample_recipe)
    assert "[菜名] 宫保鸡丁" in text
    assert "[食材]" in text
    assert "鸡胸肉" in text
    assert "[步骤]" in text
    assert "步骤 1:" in text
    assert "步骤 3:" in text
    assert "[主厨贴士] 火候要快" in text
    assert "[菜系] 川菜" in text
    assert "[难度] 中等" in text


def test_build_full_text_no_optional_fields():
    """_build_full_text should omit optional fields that are None."""
    recipe = Recipe(name="简单菜", steps=[], ingredients=[])
    text = RecipeIngestionService._build_full_text(recipe)
    assert "[菜名] 简单菜" in text
    assert "[食材]" in text
    assert "[步骤]" not in text  # steps is empty -> "步骤" header not present
    assert "[主厨贴士]" not in text


# -- chunking ----------------------------------------------------------------


def test_chunk_steps_count(sample_recipe: Recipe):
    """_chunk_steps should produce one chunk per cooking step."""
    chunks = RecipeIngestionService._chunk_steps(sample_recipe, "parent-123")
    assert len(chunks) == 3
    assert all(c.parent_id == "parent-123" for c in chunks)
    assert [c.step_number for c in chunks] == [1, 2, 3]
    assert all(c.metadata["doc_type"] == "child" for c in chunks)


def test_chunk_steps_text_content(sample_recipe: Recipe):
    """Each chunk text should contain the recipe name and step instruction."""
    chunks = RecipeIngestionService._chunk_steps(sample_recipe, "pid-1")
    assert "[菜名: 宫保鸡丁]" in chunks[0].text
    assert "步骤 1:" in chunks[0].text
    assert "加少许油" in chunks[0].text


# -- error handling ----------------------------------------------------------


def test_ingest_embedding_failure_cleans_up(ingestion_service: RecipeIngestionService, sample_recipe: Recipe):
    """If embedding fails (via vector_store add), the parent entry should be removed."""
    ingestion_service._vector_store.add_documents.side_effect = Exception("Embedding API down")
    with pytest.raises(ChromaStorageError):
        ingestion_service.ingest_recipe(sample_recipe)
    # Parent should have been cleaned up
    assert len(ingestion_service.parent_recipes) == 0


def test_ingest_chroma_add_failure(ingestion_service: RecipeIngestionService, sample_recipe: Recipe):
    """If Chroma add fails, parent is cleaned up."""
    ingestion_service._vector_store.add_documents.side_effect = Exception("Chroma disk full")
    with pytest.raises(ChromaStorageError):
        ingestion_service.ingest_recipe(sample_recipe)
    assert len(ingestion_service.parent_recipes) == 0


def test_parent_persistence_roundtrip(ingestion_service: RecipeIngestionService, sample_recipe: Recipe, tmp_path):
    """save_parents -> load_parents should recover all parent data."""
    ingestion_service.ingest_recipe(sample_recipe)

    save_path = tmp_path / "parents.json"
    ingestion_service.save_parents(str(save_path))

    # New service instance with fresh mocks
    new_embeddings = MagicMock(spec=OpenAIEmbeddings)
    new_vector_store = MagicMock(spec=Chroma)
    new_service = RecipeIngestionService(embeddings=new_embeddings, vector_store=new_vector_store)
    new_service.load_parents(str(save_path))

    assert len(new_service.parent_recipes) == 1
    recovered = list(new_service.parent_recipes.values())[0]
    assert recovered.recipe_name == "宫保鸡丁"
    assert "鸡胸肉" in recovered.full_text
