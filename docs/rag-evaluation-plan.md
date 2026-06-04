# RAG 评测与优化计划

## 背景

当前 RAG 管线：
- **检索**: 三路（Chroma 向量 + Parent-Child 富化 + BM25 关键词）+ RRF 融合
- **生成**: Moonshot LLM 流式输出，RAG / 对话双模式提示词
- **现状**: DashScope 免费额度耗尽，向量检索不可用，目前仅 BM25 + LLM 自身知识工作

本计划涵盖 Recall 评测、消融实验和优化方向，按优先级排列。

---

## 第一阶段：评测基础设施搭建

### 1.1 构建评测数据集

目标是构建一个包含 **查询 — 相关文档 ID** 标注的评测集。

**方法 A：半自动生成（推荐，省力）**
1. 从 `dishes/` 中选 50~80 个代表性菜谱，覆盖各菜系
2. 对每个菜谱用 LLM 生成 2~3 个用户查询（自然语言问法）
3. 人工验证查询合理性和文档关联性
4. 标注格式示例：

```json
{
  "query": "宫保鸡丁需要哪些食材？",
  "relevant_parent_ids": ["abc123"],
  "cuisine": "川菜"
}
```

**方法 B：人工构建（更精确）**
- 手工编写 100+ 条查询和对应相关文档
- 适合正式发表或严格对比评测

**工具脚本建议：**
```
backend/eval/
├── build_dataset.py         # 半自动生成 Q&A pair
├── dataset.json             # 评测数据集
└── metrics.py               # 评测指标实现
```

### 1.2 评测指标实现

需要在 `backend/eval/metrics.py` 中实现以下指标：

#### 检索指标 (Retrieval Metrics)

| 指标 | 含义 | 公式 |
|------|------|------|
| **Recall@K** | 前 K 个结果中包含相关文档的比例 | `relevant_retrieved / total_relevant` |
| **Precision@K** | 前 K 个结果中相关文档的比例 | `relevant_retrieved / K` |
| **MRR** | 第一个相关文档的倒数排名 | `1 / rank_first_relevant` 的均值 |
| **NDCG@K** | 归一化折损累计增益 | 标准信息检索 NDCG |
| **F1@K** | Recall 和 Precision 的调和平均 | `2 * P * R / (P + R)` |

#### 生成指标 (Generation Metrics)

| 指标 | 含义 | 方法 |
|------|------|------|
| **Faithfulness** | 回答是否忠实于上下文 | LLM-as-judge 打分 (1-5) |
| **Answer Relevance** | 回答是否针对用户问题 | LLM-as-judge 打分 (1-5) |
| **Hallucination Rate** | 编造内容的比例 | 检测是否引用了不存在的菜谱/步骤 |

---

## 第二阶段：Recall 评测实验

### 2.1 完整管线 Recall 评测

**目标**: 测量当前 3 路检索 + RRF 融合的召回能力

| 参数 | 默认值 |
|------|--------|
| vector_top_k | 5 |
| bm25_top_k | 5 |
| RRF k | 60 |
| final_top_n | 3 |

运行方式：
```bash
cd backend
python -m eval.run_recall_eval
```

输出: `Recall@1,3,5, MRR, NDCG@3, Precision@3`

### 2.2 消融实验

#### 实验 A：检索通道消融

| 实验编号 | 配置 | 意义 |
|---------|------|------|
| A1 (baseline) | 向量 + BM25 + RRF | 当前完整管线 |
| A2 | 仅向量搜索 | 评估向量单独贡献 |
| A3 | 仅 BM25 | 评估关键词单独贡献 |
| A4 | 向量 + BM25 (无 RRF) | 评估 RRF 融合增益 |
| A5 | 向量 + RRF | 评估 BM25 的增益 |

**注意**: 当前 DashScope 额度耗尽，A1/A2/A4/A5 暂不可测。可先从 A3 开始，待 embedding 恢复后补全。

#### 实验 B：RRF 参数消融

| 实验编号 | RRF k | final_top_n | 说明 |
|---------|--------|-------------|------|
| B1 | 10 | 3 | 激进融合 |
| B2 | 30 | 3 | 中等融合 |
| B3 | 60 | 3 | 当前默认 |
| B4 | 100 | 3 | 保守融合 |
| B5 | 60 | 5 | 更多候选 |
| B6 | 60 | 1 | 精确匹配 |

#### 实验 C：检索数量消融

| 实验编号 | vector_top_k | bm25_top_k | 说明 |
|---------|-------------|------------|------|
| C1 | 5 | 5 | 当前默认 |
| C2 | 10 | 10 | 更多候选 |
| C3 | 3 | 10 | 向量少+BM25多 |
| C4 | 10 | 3 | BM25少+向量多 |
| C5 | 20 | 20 | 大量候选 |

### 2.3 消融实验脚本设计

```python
# eval/run_ablation.py 核心结构

configs = [
    {"name": "A1_full", "vector_top_k": 5, "bm25_top_k": 5, "rrf_k": 60, "final_top_n": 3},
    {"name": "A3_bm25_only", "vector_top_k": 0, "bm25_top_k": 5, "rrf_k": 60, "final_top_n": 3},
    {"name": "B1_rrf_k10", "vector_top_k": 5, "bm25_top_k": 5, "rrf_k": 10, "final_top_n": 3},
    # ...
]

results = []
for cfg in configs:
    svc = RAGService(...)
    metrics = evaluate(svc, dataset, cfg)
    results.append({"config": cfg["name"], **metrics})

# 输出对比表格
print_table(results)
```

---

## 第三阶段：生成质量评测

### 3.1 LLM-as-Judge 评估

用强 LLM（如 GPT-4 / Claude）对生成结果打分：

```bash
cd backend
python -m eval.run_gen_eval --judge gpt-4
```

评估维度：
1. **Faithfulness** (1-5): 回答是否基于检索到的菜谱信息
2. **Relevance** (1-5): 回答是否切题
3. **Completeness** (1-5): 是否覆盖了查询需要的全部信息
4. **Conciseness** (1-5): 是否简洁无冗余

### 3.2 带检索 vs 不带检索对比

| 实验 | 检索 | 提示词 | 目的 |
|------|------|--------|------|
| D1 | ✅ RAG 模式 | RAG_SYSTEM_PROMPT | 完整 RAG (需 embedding 恢复) |
| D2 | ❌ 纯对话 | CHAT_SYSTEM_PROMPT | 纯 LLM 知识 |
| D3 | ✅ BM25 only | RAG_SYSTEM_PROMPT | 当前状态 |

对比指标：Faithfulness, Hallucination Rate

---

## 第四阶段：优化方向

### 4.1 紧急修复（DashScope 额度）

**选项 A**：更换 Embedding 模型
```python
# embedder.py 配置新供应商
# 可选：SiliconFlow (免费)、OpenAI (付费)、本地 bge-small (免费)
```

**选项 B**：在 DashScope 控制台开通付费
- 错误信息：`disable the "use free tier only" mode in the management console`

**选项 C**：纯 BM25 模式（当前变通方案）
- 可使用 BM25 only + LLM 知识组合
- 在 `generate_answer_stream` 中跳过 embedding 步骤即可

### 4.2 分块策略优化

| 策略 | 方法 | 预期效果 |
|------|------|---------|
| 当前（按步骤） | 每步一个 chunk | 细粒度，但语义碎片化 |
| 按段落 | 合并连续步骤 | 更多上下文，减少碎片 |
| 滑动窗口 | 2-3 步重叠窗口 | 兼顾细粒度和上下文 |
| 语义分块 | 基于 embedding 相似度切分 | 更自然的分割 |

实验建议：
```
backend/eval/chunk_strategies/
├── by_step.py    (当前)
├── by_section.py (按段落分组)
├── sliding.py    (滑动窗口)
└── compare.py    (对比脚本)
```

### 4.3 重排序优化

在 RRF 之后增加 Cross-encoder 重排序层：

```
Route 1 (向量) ─┐
                 ├─→ RRF 融合 ─→ Cross-encoder ─→ 最终结果
Route 3 (BM25) ─┘
```

可选模型：
- `BAAI/bge-reranker-v2-m3` (免费，本地运行)
- `mixedbread-ai/mxbai-rerank-xsmall-v1`

### 4.4 混合检索权重调优

当前 RRF 对向量和 BM25 等价对待。可引入加权 WRF：

```python
# WRF: score(p) = w1/(k + rank_vector) + w3/(k + rank_bm25)
def wrf_fuse(vector_results, bm25_results, w1=0.5, w3=0.5, ...):
    ...
```

实验：grid search w1 ∈ {0.2, 0.5, 0.8}, w3 = 1 - w1

### 4.5 提示词优化

当前 RAG 提示词约 500 字中文。可测试：
1. 精简版（200 字）：减少规则，只保留核心约束
2. 结构化输出版：要求以固定格式输出（"食材：xxx\n步骤：xxx"）
3. Few-shot 版：在 prompt 中加入示例 Q&A

---

## 实验矩阵总览

| 优先级 | 实验 | 依赖 | 工作量 | 影响 |
|--------|------|------|--------|------|
| P0 | 数据集构建 | 无 | 2-3h | 基础 |
| P0 | Recall 指标实现 | 数据集 | 1h | 基础 |
| P0 | BM25 消融 (A3) | 数据集 | 1h | 高 |
| P1 | DashScope 修复 | 付费/换模型 | 0.5h | 高 |
| P1 | 完整管线 Recall (A1) | DashScope | 1h | 高 |
| P1 | RRF 参数消融 (B系列) | 数据集 | 2h | 中 |
| P2 | 生成质量评测 (D系列) | 数据集 | 2h | 中 |
| P2 | 分块策略对比 | 数据集 | 3h | 中 |
| P3 | Cross-encoder 重排序 | 无 | 4h | 中 |
| P3 | WRF 加权融合 | 数据集 | 2h | 低 |

---

## 快速开始

### 1. 创建评测目录结构

```
backend/eval/
├── __init__.py
├── dataset.py           # 数据集加载 + 标注格式
├── metrics.py           # Recall/Precision/MRR/NDCG
├── run_recall_eval.py   # 检索评测入口
├── run_ablation.py      # 消融实验入口
├── run_gen_eval.py      # 生成质量评测
└── dataset.json         # 评测数据集
```

### 2. 评测数据格式

```json
[
  {
    "id": "q001",
    "query": "宫保鸡丁怎么做？",
    "relevant_parent_ids": ["<parent_id>"],
    "cuisine": "川菜"
  },
  {
    "id": "q002",
    "query": "推荐几个清淡的汤",
    "relevant_parent_ids": ["<id1>", "<id2>"],
    "cuisine": "汤类"
  }
]
```

### 3. 单次评测运行

```python
# 伪代码
dataset = load_dataset("eval/dataset.json")
svc = RAGService(settings=Settings())

for item in dataset:
    results = svc.retrieve(item["query"])
    recall = recall_at_k(results, item["relevant_parent_ids"], k=3)
    mrr = mrr(results, item["relevant_parent_ids"])
    print(f"{item['id']}: Recall@3={recall:.2f}, MRR={mrr:.2f}")

print_summary()
```
