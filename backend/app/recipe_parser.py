"""
Recipe Markdown Parser — converts HowToCook-style .md files into Recipe models.

The expected markdown format::

    # 菜名的做法

    预估烹饪难度：★★★

    ## 必备原料和工具

    - ingredient name
    - ...

    ## 计算

    - ingredient name amount unit
    - ...

    ## 操作

    - Step description
    - ...

    ## 附加内容

    - Notes and tips
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

from app.models.recipe import (
    CookingStep,
    Ingredient,
    Recipe,
    Seasoning,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUISINE_MAP = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调味料",
    "dessert": "甜品",
    "drink": "饮品",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤类",
    "staple": "主食",
    "vegetable_dish": "素菜",
}

COMMON_SEASONINGS: set[str] = {
    "盐", "酱油", "老抽", "生抽", "醋", "陈醋", "香醋", "白醋",
    "料酒", "糖", "白糖", "冰糖", "蜂蜜", "饴糖",
    "香油", "麻油", "辣椒油", "花椒油", "蒜油",
    "蚝油", "味精", "味素", "鸡精", "鸡粉",
    "胡椒粉", "黑胡椒", "白胡椒", "花椒粉", "辣椒粉",
    "五香粉", "十三香", "花椒", "八角", "桂皮", "香叶", "草果",
    "豆瓣酱", "甜面酱", "番茄酱", "辣椒酱", "蒜蓉辣酱",
    "豆豉", "腐乳", "料酒", "米酒",
    "淀粉", "玉米淀粉", "红薯淀粉", "生粉",
    "小苏打", "泡打粉", "酵母",
}

# Regex to capture amount + unit from ingredient lines, e.g. "115g", "2 个", "10-15ml"
AMOUNT_RE = re.compile(
    r"(\d+(?:[~\-]\d+)?(?:\.\d+)?)\s*"
    r"(克|毫升|ml|ML|g|G|个|只|根|把|汤匙|茶匙|小勺|大勺|"
    r"勺|滴|片|条|袋|盒|瓶|包|份|斤|两|cm|厘米|mm|毫米|"
    r"升|l|L|kg|KG|粒|段|瓣|块|束|撮|碗|杯|罐|盘|扎|"
    r"寸|英寸|cm|厘米)"
)

# Regex for the difficulty line: 预估烹饪难度：★★★★
DIFFICULTY_RE = re.compile(r"预估烹饪难度[：:]\s*([★☆]+)")

# Regex for section headers: ## 名称
SECTION_HEADER_RE = re.compile(r"^##\s+(.+)", re.MULTILINE)

# ---------------------------------------------------------------------------
# Text Helpers
# ---------------------------------------------------------------------------


def _strip_html_comments(text: str) -> str:
    """Remove HTML comments (``<!-- ... -->``) from the text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _strip_markdown_formatting(text: str) -> str:
    """Remove bold/italic markers (``**``, ``__``) from inline text."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


# ---------------------------------------------------------------------------
# Section Extraction
# ---------------------------------------------------------------------------


def _extract_sections(content: str) -> dict[str, str]:
    """Split the markdown content by ``## SectionHeader`` boundaries.

    Returns a dict mapping section title → body text (with any sub-headers
    included in the body).
    """
    sections: dict[str, str] = {}
    # Find all "## Title" positions
    matches = list(SECTION_HEADER_RE.finditer(content))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        sections[title] = body
    return sections


def _get_section(sections: dict[str, str], *names: str) -> str:
    """Return the first section matching any of ``names``, or ``""``."""
    for name in names:
        if name in sections:
            return sections[name]
    return ""


# ---------------------------------------------------------------------------
# Ingredient Parsing
# ---------------------------------------------------------------------------


def _parse_bullet_item(line: str) -> str:
    """Strip list prefix from a line: ``- 土豆`` → ``土豆``."""
    line = line.strip()
    # Remove leading -, *, or 1. etc.
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^\d+\.\s+", "", line)
    # Remove trailing parenthetical notes like （推荐品牌好侍）
    line = re.sub(r"\s*[（(].*[）)]", "", line).strip()
    return line


def _parse_bullet_lines(text: str) -> list[str]:
    """Extract bullet items from a section body.

    Handles ``- `` and ``1. `` prefixed lines.
    """
    items: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.\s+", line):
            items.append(line)
    return items


def _parse_amount(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse ``"土豆 2 个（约240g）"`` → ``("土豆", "2", "个")``.

    Returns ``(name, amount, unit)`` — amount / unit are ``None`` when
    no quantity is found.
    """
    # Remove parenthetical notes first for amount parsing
    cleaned = re.sub(r"\s*[（(].*[）)]", "", text).strip()

    m = AMOUNT_RE.search(cleaned)
    if m:
        amount_str = m.group(1)
        unit_str = m.group(2)
        # Name is everything before the amount match
        name = cleaned[: m.start()].strip()
        return name, amount_str, unit_str

    # No amount found — the whole thing is the name
    return text.strip(), None, None


def _classify_item(name: str, seasonings_set: set[str]) -> bool:
    """Return ``True`` if *name* is a seasoning."""
    # Check the full name or individual tokens
    if name in seasonings_set:
        return True
    # Check comma-separated parts (e.g. "葱、姜、蒜")
    for part in re.split(r"[、,，]", name):
        if part.strip() in seasonings_set:
            return True
    return False


def _merge_ingredients(
    names_section: str, calc_section: str, seasonings_set: set[str]
) -> tuple[list[Ingredient], list[Seasoning]]:
    """Merge ingredient data from ``必备原料和工具`` and ``计算`` sections.

    Returns ``(ingredients, seasonings)``.
    """
    # 1. Parse names from "必备原料" section
    name_only_items = []
    for line in _parse_bullet_lines(names_section):
        item = _parse_bullet_item(line)
        if item:
            name_only_items.append(item)

    # 2. Parse name+amount from "计算" section
    calc_map: dict[str, tuple[Optional[str], Optional[str]]] = {}
    calc_items = []
    for line in _parse_bullet_lines(calc_section):
        raw_name = _parse_bullet_item(line)
        name, amount, unit = _parse_amount(line)
        # Use the parsed name if available, fallback to raw
        final_name = name or raw_name
        calc_map[final_name] = (amount, unit)
        calc_items.append(final_name)

    # 3. Merge: calc overrides names, supplemented by names-only items
    seen: set[str] = set()
    merged_names: list[str] = []
    for item in calc_items:
        if item not in seen:
            merged_names.append(item)
            seen.add(item)
    for item in name_only_items:
        if item not in seen:
            merged_names.append(item)
            seen.add(item)

    # 4. Classify each item
    ingredients: list[Ingredient] = []
    seasonings_list: list[Seasoning] = []
    for name in merged_names:
        amount, unit = calc_map.get(name, (None, None))
        if _classify_item(name, seasonings_set):
            seasonings_list.append(Seasoning(name=name, amount=amount, unit=unit))
        else:
            ingredients.append(Ingredient(name=name, amount=amount, unit=unit))

    return ingredients, seasonings_list


# ---------------------------------------------------------------------------
# Step Parsing
# ---------------------------------------------------------------------------


def _parse_steps(text: str) -> list[CookingStep]:
    """Parse the ``操作`` section body into ``CookingStep`` objects."""
    steps: list[CookingStep] = []
    lines = _parse_bullet_lines(text)

    for idx, line in enumerate(lines, start=1):
        instruction = _parse_bullet_item(line)
        instruction = _strip_markdown_formatting(instruction)

        # Try to extract a tip from parentheticals at the end
        tip: Optional[str] = None
        tip_match = re.search(r"[（(](.*?贴士[^）)]*)[）)]", instruction)
        if tip_match:
            tip = tip_match.group(1)
            # Remove the tip from the instruction
            instruction = instruction.replace(f"（{tip}）", "").replace(f"({tip})", "").strip()

        steps.append(
            CookingStep(
                step_number=idx,
                instruction=instruction,
                tips=tip,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Difficulty Parsing
# ---------------------------------------------------------------------------


def _parse_difficulty(content: str) -> Optional[str]:
    """Extract difficulty level, e.g. ``"★★★★"``."""
    m = DIFFICULTY_RE.search(content)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Name Parsing
# ---------------------------------------------------------------------------


def _parse_dish_name(content: str) -> Optional[str]:
    """Extract dish name from ``# 菜名的做法``."""
    m = re.search(r"^#\s+(.+?)(?:的做法)?\s*$", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_recipe_file(filepath: str, cuisine_type: Optional[str] = None) -> Optional[Recipe]:
    """Parse a single .md recipe file into a ``Recipe`` model.

    Returns ``None`` if the file cannot be parsed (missing title, etc.).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to read %s: %s", filepath, exc)
        return None

    # Strip HTML comments
    content = _strip_html_comments(raw)

    # Dish name
    name = _parse_dish_name(content)
    if not name:
        return None

    # Difficulty
    difficulty = _parse_difficulty(content)

    # Sections
    sections = _extract_sections(content)
    names_sec = _get_section(sections, "必备原料和工具")
    calc_sec = _get_section(sections, "计算")
    oper_sec = _get_section(sections, "操作")
    notes_sec = _get_section(sections, "附加内容")

    # Steps
    steps = _parse_steps(oper_sec)

    # Ingredients & Seasonings
    ingredients, seasonings = _merge_ingredients(names_sec, calc_sec, COMMON_SEASONINGS)

    # Global tips
    global_tips: Optional[str] = None
    if notes_sec:
        # Strip list markers from notes
        lines = []
        for line in notes_sec.split("\n"):
            line = line.strip()
            if line:
                lines.append(re.sub(r"^[-*\d.]+\s*", "", line))
        if lines:
            global_tips = "\n".join(lines)

    return Recipe(
        name=name,
        ingredients=ingredients,
        seasonings=seasonings,
        steps=steps,
        cuisine_type=cuisine_type,
        difficulty=difficulty,
        global_tips=global_tips,
    )


def parse_all_recipes(dishes_dir: str) -> list[Recipe]:
    """Recursively find and parse all ``.md`` files under *dishes_dir*.

    Cuisine type is inferred from the immediate parent directory name.
    """
    recipes: list[Recipe] = []
    pattern = os.path.join(dishes_dir, "**", "*.md")
    for filepath in sorted(glob.glob(pattern, recursive=True)):
        # Skip template files
        if "template" in filepath.replace("\\", "/").split("/"):
            continue

        # Infer cuisine type from parent directory
        parent_dir = os.path.basename(os.path.dirname(filepath))
        cuisine = CUISINE_MAP.get(parent_dir)

        recipe = parse_recipe_file(filepath, cuisine_type=cuisine)
        if recipe is None:
            continue

        recipes.append(recipe)

    return recipes
