from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    """A single ingredient with optional amount/unit."""

    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None


class Seasoning(BaseModel):
    """A seasoning / condiment."""

    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None


class CookingStep(BaseModel):
    """A single cooking step with optional duration and tips."""

    step_number: int
    instruction: str
    duration_minutes: Optional[int] = None
    tips: Optional[str] = None


class Recipe(BaseModel):
    """Full recipe input from the user / data pipeline."""

    name: str
    ingredients: list[Ingredient] = Field(default_factory=list)
    seasonings: list[Seasoning] = Field(default_factory=list)
    steps: list[CookingStep] = Field(default_factory=list)
    cuisine_type: Optional[str] = None
    difficulty: Optional[str] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    global_tips: Optional[str] = None


class RecipeParent(BaseModel):
    """Full parent recipe stored in-memory."""

    parent_id: str
    recipe_name: str
    full_text: str
    raw_data: dict


class ChildChunk(BaseModel):
    """A single chunked step ready for embedding."""

    chunk_id: str
    parent_id: str
    text: str
    step_number: int
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Unified search result produced by the retrieval pipeline."""

    parent_id: str
    recipe_name: str
    text: str
    score: float
    source: str  # "vector" | "bm25" | "vector+bm25"
    rank_vector: Optional[int] = None
    rank_bm25: Optional[int] = None


def generate_id() -> str:
    """Generate a unique string ID."""
    return uuid4().hex
