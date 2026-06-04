# 架构评审报告：RAG 系统智能体能力升级

## 评审背景

基于现有代码库状态进行评审：
- `app/core/rag/` — 11 个原子模块，4 个编排类，530 行 `engine.py`
- 40 个单元测试，全部通过
- 基础 RAG（向量 + BM25 + RRF）运转良好
- 当前入口：`RAGService` 门面类，`StreamingGenerator.generate_answer_stream()` 输出

升级目标：**语义路由 + 工具整合**。需要支持三类用户请求：

| 意图 | 示例 | 动作 |
|------|------|------|
| `chat` | "你好" / "你是谁" | 直接 LLM 回复，不查 RAG |
| `recipe` | "宫保鸡丁怎么做" | 走完整 RAG 管线 |
| `health` | "200g 鸡胸肉多少卡路里" | 提取参数 → 调用函数 → 结果反哺 RAG |

---

## 路线一：原生 Python 线性条件流控

### 核心架构设计

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ IntentClassifier                                       │
│  ├─ LLM 单轮分类（推荐）或 关键词启发式                 │
│  └─ 返回: {intent: str, params: dict}                  │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐
    │ chat    │ │ recipe   │ │ health   │
    │ handler │ │ handler  │ │ handler  │
    └─────────┘ └──────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       ▼
               LLM 流式生成 + 上下文
```

### 在你现有架构中的拼装方式

核心改动点在 `RAGService` 和 `StreamingGenerator`。现有文件**零删除，只新增**：

```
app/core/rag/
├── intent_classifier.py    # 新增：意图分类 + 参数提取
├── tool_registry.py        # 新增：工具注册表
├── tools/
│   ├── __init__.py
│   └── calculate_calories.py  # 新增：热量计算工具
└── engine.py               # 小改：RAGService 增加路由入口
```

核心拼接代码：

```python
# app/core/rag/intent_classifier.py

from pydantic import BaseModel

class IntentResult(BaseModel):
    intent: str           # "chat" | "recipe" | "health"
    confidence: float
    params: dict = {}     # health 场景: {"ingredient": "鸡胸肉", "weight_g": 200}

class IntentClassifier:
    """用 LLM 单次调用做意图分类 + 参数提取。

    为什么不走关键词匹配？
    因为 "200g 鸡胸肉多少热量" 需要提取数值和单位，
    关键词搞不定 "一份"、"半斤"、"大约300克" 这种自然语言表达。
    """

    SYSTEM_PROMPT = (
        "你是一个意图分类器。判断用户输入属于以下三类之一：\n"
        "1. chat — 闲聊、打招呼、问身份等，不需要查菜谱\n"
        "2. recipe — 询问菜谱做法、食材、步骤等，需要检索菜谱库\n"
        "3. health — 询问热量、营养、健康建议、身体数据计算\n\n"
        "如果是 health 意图，请提取参数到 params 中：\n"
        "  - ingredient: 食材名称\n"
        "  - weight_g: 重量（克），请将斤/两/公斤等统一转为克\n\n"
        "以 JSON 格式回复：{\"intent\": \"...\", \"confidence\": 0.95, \"params\": {...}}"
    )

    def classify(self, query: str, llm: ChatOpenAI) -> IntentResult:
        import json
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("human", query),
        ]
        # 非流式调用，一次出结果
        response = llm.invoke(messages).content
        # 解析 JSON，带 fallback
        try:
            data = json.loads(response)
            return IntentResult(**data)
        except (json.JSONDecodeError, ValidationError):
            return IntentResult(intent="recipe", confidence=0.5, params={})
```

```python
# app/core/rag/tools/calculate_calories.py

def calculate_calories(ingredient: str, weight_g: float) -> dict:
    """查询食材热量数据库，返回热量信息。

    当前是本地硬编码数据，后续可对接外部 API 或数据库。
    """
    CALORIE_DB = {
        "鸡胸肉": 1.65,   # kcal/g（生）
        "鸡腿肉": 1.85,
        "猪瘦肉": 1.43,
        "牛肉": 2.50,
        "三文鱼": 2.08,
        "巴沙鱼": 0.92,
        "鸡蛋": 1.55,      # per 50g
        "豆腐": 0.76,
        "米饭": 1.30,      # 熟
    }
    kcal_per_g = CALORIE_DB.get(ingredient)
    if kcal_per_g is None:
        return {
            "ingredient": ingredient,
            "weight_g": weight_g,
            "calories_kcal": None,
            "warning": f"未找到「{ingredient}」的热量数据",
        }
    total = round(kcal_per_g * weight_g, 1)
    return {
        "ingredient": ingredient,
        "weight_g": weight_g,
        "calories_kcal": total,
        "per_100g": round(kcal_per_g * 100, 1),
    }
```

```python
# 在 engine.py 中 — RAGService 新增路由方法

class RAGService:
    # ... 现有代码保持不变 ...

    # --- 新增：路由分类器（懒初始化） ---
    @cached_property
    def _classifier(self) -> IntentClassifier:
        return IntentClassifier()

    # --- 新增：统一入口 ---
    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """统一的用户查询入口。自动路由到 chat / recipe / health。"""
        # Step 1: 意图分类（同步调用，一次 LLM round-trip）
        result = self._classifier.classify(query, self._llm)

        if result.intent == "chat":
            async for token in self._direct_chat(query):
                yield token
            return

        # Step 2: health 意图 → 先计算，结果作为 RAG 的增强上下文
        health_context = ""
        if result.intent == "health" and result.params:
            from app.core.rag.tools.calculate_calories import calculate_calories
            calc_result = calculate_calories(**result.params)
            health_context = self._build_health_context(calc_result)
            # 热量计算后，仍然走 RAG 检索相关食谱
            # 例如："鸡胸肉 200g 约 330kcal。以下是低卡鸡胸肉食谱推荐："

        # Step 3: recipe / health 都走 RAG
        async for token in self._generator.generate_answer_stream(
            query, extra_context=health_context
        ):
            yield token

    def _build_health_context(self, calc_result: dict) -> str:
        kcal = calc_result.get("calories_kcal")
        if kcal is None:
            return f"用户查询食材热量：{calc_result.get('warning')}"
        return (
            f"[热量计算结果]\n"
            f"食材：{calc_result['ingredient']}\n"
            f"重量：{calc_result['weight_g']}g\n"
            f"热量：{kcal}kcal（每100g约 {calc_result['per_100g']}kcal）\n"
        )
```

注意：`StreamingGenerator.generate_answer_stream()` 需要增加一个可选的 `extra_context` 参数，追加到 prompt 中。这个改动 3 行代码。

### 工程上限分析

**硬伤：if-else 链不会爆炸，前提是你用 registry。**

如果直接用 `if intent == 'health'` 堆 10 个分支，确实会崩。但用 **注册表 + 策略模式** 可以优雅地撑到至少 20-30 个工具：

```python
# tool_registry.py

from typing import Protocol, AsyncGenerator

class ToolHandler(Protocol):
    """工具协议：每个工具就是一个函数，输入 → 输出上下文字符串。"""
    async def execute(self, params: dict) -> str: ...

class ToolRegistry:
    """工具注册表：intent → handler 映射。"""
    def __init__(self):
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, intent: str, handler: ToolHandler) -> None:
        self._handlers[intent] = handler

    def get(self, intent: str) -> ToolHandler | None:
        return self._handlers.get(intent)
```

这样新增一个工具 = 写 handler 函数 + `registry.register("new_intent", handler)`。不需要碰任何 if-else。

**真正的上限在哪里？**

1. **意图分类是瓶颈**：当意图达到 15+ 时，单次 LLM 分类的准确率会下降。届时需要升级分类策略（few-shot 或专用分类模型）。
2. **无自动推理**：如果用户问的是一个需要**链式调用**多个工具的问题（"200g 鸡胸肉多少热量，然后推荐一个相关低卡菜谱"），线性流程需要硬编码编排。Agent 的优势在这里。
3. **工具间无状态共享**：工具的执行结果只能通过 `extra_context` 字符串传给下游，无法做结构化的状态共享。

### 总结路线一

```
优点：  零新依赖 / 完全可观测可断点 / 测试方法与现有 40 个测试完全一致
       / 与现有代码风格一致 / TTFT 最低（分类 1 次 LLM + 生成 1 次流）
缺点：  15+ 意图后分类准确率下降 / 无自动链式推理 / 多工具编排需手写
工程上限：可支撑 20-30 个工具，前提是用注册表替代 if-else
```

---

## 路线二：LangChain Agent（AgentExecutor / ReAct）

### 对接方案

LangChain Agent 的核心思路是：把你的 RAG 管线包装成一个 `BaseTool`，让 Agent 自主决定何时调用它。

```python
# 方案二示例：将现有管线包装为 LangChain Tool

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

class RAGPipelineTool(BaseTool):
    """将你现有的三路召回 RAG 管线包装为 Tool。

    关键设计决策：保留内部 RRF fusion，对外只暴露 query → answer string。
    """
    name = "recipe_rag"
    description = (
        "当用户询问菜谱做法、食材、步骤时使用此工具。"
        "输入应为菜名或烹饪问题。返回结构化菜谱信息。"
    )

    def __init__(self, rag_service: RAGService):
        super().__init__()
        self._rag = rag_service

    def _run(self, query: str) -> str:
        """同步运行（AgentExecutor 要求同步）。"""
        # 这里需要用 event loop 桥接异步的 generate_answer_stream
        loop = asyncio.new_event_loop()
        collected = []
        async for token in self._rag.generate_answer_stream(query):
            collected.append(token)
        return "".join(collected)

    async def _arun(self, query: str) -> str:
        collected = []
        async for token in self._rag.generate_answer_stream(query):
            collected.append(token)
        return "".join(collected)


class CalorieTool(BaseTool):
    """热量计算工具。"""
    name = "calculate_calories"
    description = (
        "当用户询问食材热量、卡路里时使用此工具。"
        "输入格式应为：食材名称,重量(克)。例如：鸡胸肉,200"
    )

    def _run(self, input_str: str) -> str:
        ingredient, weight_str = input_str.split(",")
        result = calculate_calories(ingredient.strip(), float(weight_str.strip()))
        kcal = result.get("calories_kcal")
        if kcal is None:
            return f"未找到「{ingredient}」的热量数据。"
        return f"{result['ingredient']} {result['weight_g']}g 约 {kcal}kcal（每100g约 {result['per_100g']}kcal）"


# ---- 在 RAGService 中组装 Agent ----

class RAGService:
    # ... 现有代码 ...

    def _build_agent(self) -> AgentExecutor:
        tools = [
            RAGPipelineTool(self),
            CalorieTool(),
        ]
        prompt = PromptTemplate.from_template(
            """你是一个智能烹饪助手。你有以下工具可用：
{tools}

请根据用户问题，决定是否需要使用工具。
如果需要，请按以下格式回复：
Thought: 我需要使用什么工具
Action: 工具名称
Action Input: 工具输入

如果不需要工具，直接回复即可。
...
"""
        )
        agent = create_react_agent(self._llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, ...)
```

### 对你现有 11 个原子文件的冲击

| 文件 | 影响程度 | 说明 |
|------|----------|------|
| `engine.py` (4 classes) | **中** | RAGService 需要加 AgentExecutor 初始化逻辑。RecipeRetriever 等内部不变。 |
| `embedder.py` | 无 | 仍然通过工厂函数使用 |
| `vector_store.py` | 无 | 仍然通过工厂函数使用 |
| `keyword_search.py` | 无 | 仍然通过 BM25Retriever 使用 |
| `ranker.py` | 无 | RRF 是纯函数，Tool 内部调用 |
| `prompt_manager.py` | **中** | SYSTEM_PROMPT 需要扩展以兼容 Agent 场景 |
| `data_processor.py` | 无 | 纯函数不变 |
| `config.py` | 无 | 不变 |
| `exceptions.py` | 无 | 不变 |
| `__init__.py` | 无 | 不变 |
| `rag_service.py` (facade) | 无 | 不变 |

**关键结论**：11 个原子文件中的 8 个完全不需要改动。核心受影响的是 `engine.py`（RAGService 扩展）和 `prompt_manager.py`（提示词兼容）。

### 隐藏的坑

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| **ReAct 循环多一次 LLM 调用** | ⚠️ 高 | Agent 先要"思考"是否调工具，然后才执行。这意味着每次 recipe 查询 → 最少 2 次 LLM round-trip（思考 + 生成），TTFT 翻倍 |
| **Tool 输入输出都是字符串** | ⚠️ 中 | `BaseTool` 的 `_run` 签名是 `str → str`。你精心设计的 `SearchResult` 结构体、`RRF 分数`、`rank_vector/rank_bm25` 全部被压扁成字符串。结构化信息丢失 |
| **Agent 可能跳过 RAG 直接胡编** | ⚠️ 高 | Agent 的 ReAct 循环可能错误判断"不需要工具"，直接拿自身知识回复。这与你的核心原则"严格基于参考上下文"相悖 |
| **LangChain Agent API 不稳定** | ⚠️ 中 | `create_react_agent` 在 langchain>=0.3 的签名与 0.2 不同。`AgentExecutor` 的行为在跨版本间有 breaking change。锁定版本是必须的 |
| **流式响应丢失** | ⚠️ 高 | `AgentExecutor` 的流式支持（`.astream_events`）在 0.3 中仍然不成熟。现有 `StreamingGenerator` 的逐 token 输出很难无损对接 Agent 的流式接口 |

### 总结路线二

```
优点：  工具注册是标准化接口 / 支持自动链式推理 / 意图分类由模型自主决定
       / 适合 15+ 工具的复杂场景
缺点：  ReAct 多一次 LLM 调用 → TTFT 翻倍 / 结构化数据丢失 / Agent 可能"
       不听话"跳过 RAG / 流式体验退化 / API 不稳定
工程上限：可支撑 100+ 工具，但 Debug 成本随工具数量指数级上升
```

---

## 📊 最终评审：三维度对比

| 维度 | 路线一（Python 流控） | 路线二（LangChain Agent） | 胜出 |
|------|----------------------|--------------------------|------|
| **TTFT / 延迟** | 分类 1 次 LLM + 流式生成 1 次 = **2 次** | ReAct 思考 1 次 + 可能再调工具 + 流式 = **最少 3 次** | **路线一** |
| **端到端延迟** | ~3-5s（取决于 Moonshot 速度） | ~5-10s（翻倍） | **路线一** |
| **代码可维护性** | if-else 可用注册表模式规避，40 个测试完全复用 | Agent 内部逻辑黑盒，难以单元测试 | **路线一** |
| **Debug 成本** | 可单步断点、可 print、可 trace | 需要开 langchain 的 debug 日志，难以复现 Agent 决策路径 | **路线一** |
| **意图扩展性** | 注册表模式可支撑 20-30 个工具 | 标准化 Tool 接口可支撑 100+ | **路线二** |
| **链式推理** | 需要手写编排逻辑 | 内置 ReAct 循环，自动多步推理 | **路线二** |

### 最终结论

**推荐路线一：原生 Python 流控。** 理由如下：

1. **你的项目规模不匹配 Agent 的复杂度。** 你目前只有 3 个意图，未来 6 个月内也不太可能超过 10 个。LangChain Agent 是为 50+ 工具的复杂场景设计的，引入它你现在就要承受它的所有复杂度，但享受不到它的核心优势（多步链式推理）。

2. **你是"Vibe Coding"——独自开发。** 这意味着：
   - 每次 Agent 行为异常，你需要自己读 LangChain 源码排查
   - 每次 LangChain 大版本升级，你需要自己迁移代码
   - 你现有的 40 个优秀测试对 AgentExecutor 内部几乎无能为力
   - **时间是你最稀缺的资源**，不要花在调试 Agent 的奇怪行为上

3. **你的现有架构已经为路线一铺好了路。** `RAGService` 本身就是门面模式，再加上一个 `IntentClassifier` + `ToolRegistry`，就是完整的"分类器-注册表-执行器"架构。未来哪天你真的需要 Agent 了：
   - 把 `IntentClassifier` 换成更复杂的分类模型（不需要改调用方）
   - 把 `ToolRegistry` 的协议对齐 LangChain 的 `BaseTool`（迁移路径清晰）
   - 把 `process_query` 中的线性调用换成 ReAct 循环（局部替换）

### 推荐的下一步代码基座

```
app/core/rag/
├── __init__.py
├── config.py             # ✅ 不变
├── data_processor.py     # ✅ 不变
├── embedder.py           # ✅ 不变
├── engine.py             # 🔧 RAGService 增加 process_query() + _classifier
├── exceptions.py         # ✅ 不变
├── intent_classifier.py  # 🆕 LLM 意图分类 + 参数提取
├── keyword_search.py     # ✅ 不变
├── llm.py                # ✅ 不变
├── prompt_manager.py     # 🔧 SYSTEM_PROMPT 扩展兼容 chat/health 场景
├── ranker.py             # ✅ 不变
├── tool_registry.py      # 🆕 工具注册表 + 协议定义
├── tools/
│   ├── __init__.py
│   └── calculate_calories.py  # 🆕 热量计算工具
└── vector_store.py       # ✅ 不变
```

**11 个现有文件中有 9 个完全不需要改动**，只需新增 3 个文件 + 小改 2 个文件。这是"最少侵入、最大价值"的路径。

要做的话，我现在就可以从 `intent_classifier.py` + `tool_registry.py` 开始落地。
