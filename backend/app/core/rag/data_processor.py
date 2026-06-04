"""Recipe text processing — full-text building and step chunking.

Pure functions extracted from ``RecipeIngestionService``.
"""

from __future__ import annotations

from app.models.recipe import ChildChunk, Recipe, generate_id


def build_full_text(recipe: Recipe) -> str:
    """Build a structured plain-text representation of a recipe."""
    lines: list[str] = []
    lines.append(f"[菜名] {recipe.name}")
    if recipe.cuisine_type:
        lines.append(f"[菜系] {recipe.cuisine_type}")
    if recipe.difficulty:
        lines.append(f"[难度] {recipe.difficulty}")
    if recipe.prep_time_minutes is not None:
        lines.append(f"[准备时间] {recipe.prep_time_minutes} 分钟")
    if recipe.cook_time_minutes is not None:
        lines.append(f"[烹饪时间] {recipe.cook_time_minutes} 分钟")

    lines.append("")
    lines.append("[食材]")
    for ing in recipe.ingredients:
        parts = [ing.name]
        if ing.amount:
            parts.append(ing.amount)
        if ing.unit:
            parts.append(ing.unit)
        lines.append(f"  - {' '.join(parts)}")

    if recipe.seasonings:
        lines.append("")
        lines.append("[调味料]")
        for s in recipe.seasonings:
            parts = [s.name]
            if s.amount:
                parts.append(s.amount)
            if s.unit:
                parts.append(s.unit)
            lines.append(f"  - {' '.join(parts)}")

    if recipe.steps:
        lines.append("")
        lines.append("[步骤]")
    for step in recipe.steps:
        tip = f" (贴士: {step.tips})" if step.tips else ""
        dur = f" [{step.duration_minutes} 分钟]" if step.duration_minutes is not None else ""
        lines.append(f"  步骤 {step.step_number}: {step.instruction}{dur}{tip}")

    if recipe.global_tips:
        lines.append("")
        lines.append(f"[主厨贴士] {recipe.global_tips}")

    return "\n".join(lines)


def chunk_steps(recipe: Recipe, parent_id: str) -> list[ChildChunk]:
    """Split recipe steps into one ``ChildChunk`` per step."""
    chunks: list[ChildChunk] = []
    for step in recipe.steps:
        tip_line = f"\n贴士: {step.tips}" if step.tips else ""
        dur_line = f" [{step.duration_minutes} 分钟]" if step.duration_minutes is not None else ""
        text = (
            f"[菜名: {recipe.name}]\n"
            f"步骤 {step.step_number}: {step.instruction}{dur_line}{tip_line}"
        )
        chunk = ChildChunk(
            chunk_id=generate_id(),
            parent_id=parent_id,
            text=text,
            step_number=step.step_number,
            metadata={
                "parent_id": parent_id,
                "recipe_name": recipe.name,
                "step_number": step.step_number,
                "doc_type": "child",
            },
        )
        chunks.append(chunk)
    return chunks
