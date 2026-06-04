"""pytest fixtures for RAG service tests."""

from __future__ import annotations

import pytest

from app.models.recipe import CookingStep, Ingredient, Recipe, Seasoning


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    """Return 3 sample Chinese recipes for testing."""
    return [
        Recipe(
            name="宫保鸡丁",
            ingredients=[
                Ingredient(name="鸡胸肉", amount="300", unit="克"),
                Ingredient(name="花生米", amount="50", unit="克"),
                Ingredient(name="干辣椒", amount="10", unit="个"),
                Ingredient(name="葱", amount="2", unit="根"),
                Ingredient(name="姜", amount="1", unit="块"),
                Ingredient(name="蒜", amount="3", unit="瓣"),
            ],
            seasonings=[
                Seasoning(name="酱油", amount="2", unit="汤匙"),
                Seasoning(name="醋", amount="1", unit="汤匙"),
                Seasoning(name="糖", amount="1", unit="汤匙"),
                Seasoning(name="料酒", amount="1", unit="汤匙"),
                Seasoning(name="淀粉", amount="1", unit="茶匙"),
            ],
            steps=[
                CookingStep(step_number=1, instruction="鸡胸肉切丁，用料酒、淀粉腌制15分钟", tips="腌制时加少许油可使肉质更嫩"),
                CookingStep(step_number=2, instruction="花生米小火炒至金黄，备用"),
                CookingStep(step_number=3, instruction="干辣椒剪段，葱姜蒜切末"),
                CookingStep(step_number=4, instruction="碗中调好酱汁：酱油、醋、糖、淀粉、水搅拌均匀"),
                CookingStep(step_number=5, instruction="热锅凉油，爆香干辣椒、葱姜蒜"),
                CookingStep(step_number=6, instruction="加入鸡丁翻炒至变色，倒入酱汁快速翻炒"),
                CookingStep(step_number=7, instruction="最后加入花生米翻炒均匀，出锅"),
            ],
            cuisine_type="川菜",
            difficulty="中等",
            prep_time_minutes=20,
            cook_time_minutes=15,
            global_tips="宫保鸡丁的关键在于火候要快，酱汁要提前调好。",
        ),
        Recipe(
            name="西红柿鸡蛋汤",
            ingredients=[
                Ingredient(name="西红柿", amount="2", unit="个"),
                Ingredient(name="鸡蛋", amount="3", unit="个"),
                Ingredient(name="葱", amount="1", unit="根"),
            ],
            seasonings=[
                Seasoning(name="盐", amount="适量"),
                Seasoning(name="香油", amount="少许"),
            ],
            steps=[
                CookingStep(step_number=1, instruction="西红柿切块，鸡蛋打散备用"),
                CookingStep(step_number=2, instruction="锅中烧水，水开后放入西红柿煮3分钟"),
                CookingStep(step_number=3, instruction="缓慢倒入蛋液，用筷子轻轻搅动形成蛋花"),
                CookingStep(step_number=4, instruction="加盐调味，淋香油，撒葱花出锅"),
            ],
            cuisine_type="家常菜",
            difficulty="简单",
            prep_time_minutes=5,
            cook_time_minutes=10,
            global_tips="蛋液要缓慢倒入才能形成漂亮的蛋花。",
        ),
        Recipe(
            name="巴沙鱼柳",
            ingredients=[
                Ingredient(name="巴沙鱼柳", amount="1", unit="条"),
                Ingredient(name="柠檬", amount="1", unit="个"),
                Ingredient(name="蒜", amount="3", unit="瓣"),
            ],
            seasonings=[
                Seasoning(name="盐", amount="适量"),
                Seasoning(name="黑胡椒", amount="适量"),
                Seasoning(name="橄榄油", amount="2", unit="汤匙"),
            ],
            steps=[
                CookingStep(step_number=1, instruction="巴沙鱼柳解冻，用厨房纸吸干水分"),
                CookingStep(step_number=2, instruction="两面撒上盐和黑胡椒腌制10分钟"),
                CookingStep(step_number=3, instruction="蒜切片，柠檬切片备用"),
                CookingStep(step_number=4, instruction="平底锅热油，放入鱼柳煎至两面金黄"),
                CookingStep(step_number=5, instruction="摆盘，放上柠檬片和蒜片装饰"),
            ],
            cuisine_type="西餐",
            difficulty="简单",
            prep_time_minutes=10,
            cook_time_minutes=15,
            global_tips="巴沙鱼柳不要煎太久，否则口感会变柴。",
        ),
    ]


@pytest.fixture
def sample_recipe() -> Recipe:
    """Return a single sample recipe (宫保鸡丁)."""
    return Recipe(
        name="宫保鸡丁",
        ingredients=[
            Ingredient(name="鸡胸肉", amount="300", unit="克"),
            Ingredient(name="花生米", amount="50", unit="克"),
        ],
        seasonings=[
            Seasoning(name="酱油", amount="2", unit="汤匙"),
            Seasoning(name="醋", amount="1", unit="汤匙"),
        ],
        steps=[
            CookingStep(step_number=1, instruction="鸡胸肉切丁腌制", tips="加少许油"),
            CookingStep(step_number=2, instruction="花生米炒至金黄"),
            CookingStep(step_number=3, instruction="爆香调料，加入鸡丁翻炒"),
        ],
        cuisine_type="川菜",
        difficulty="中等",
        global_tips="火候要快。",
    )
