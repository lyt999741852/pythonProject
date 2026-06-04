# Vibe Cooking 🍳 — AI 智能烹饪助手

Vibe Cooking 是一个基于 **RAG (Retrieval-Augmented Generation)** 架构的智能烹饪助手，拥有 **322+ 道中文菜谱** 的知识库。用户可以通过自然语言对话，获得精准的菜谱推荐、烹饪步骤指导和厨艺知识解答。

## 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | React 18 + TypeScript + Vite 6 + Tailwind CSS 3 |
| **后端** | Python 3.10+ / FastAPI + Uvicorn |
| **向量检索** | LangChain + ChromaDB + DashScope text-embedding-v4 |
| **关键词检索** | BM25 (rank-bm25) + jieba 中文分词 |
| **融合排序** | RRF (Reciprocal Rank Fusion) |
| **LLM** | Moonshot / Kimi (moonshot-v1-8k) |
| **通信** | SSE (Server-Sent Events) 流式响应 |

## 系统架构

```
┌─────────────┐     SSE Stream      ┌─────────────────────────────────────┐
│   React     │ ◄────────────────── │         FastAPI Backend             │
│   Frontend  │                     │                                     │
│  :5173      │ ── POST /api/chat ─►│  ┌────────────┐  ┌──────────────┐  │
└─────────────┘                     │  │RAGService  │  │IngestionEngine│  │
                                    │  │            │  │              │  │
                                    │  │ ┌────────┐ │  │ ┌──────────┐ │  │
                                    │  │ │Hybrid  │ │  │ │ Recipe   │ │  │
                                    │  │ │Retriever│ │  │ │ Parser   │ │  │
                                    │  │ └──┬──┬──┘ │  │ └──────────┘ │  │
                                    │  │    │  │    │  └──────────────┘  │
                                    │  │ ┌──▼──▼──┐ │                     │
                                    │  │ │LLM     │ │                     │
                                    │  │ │Chain   │ │                     │
                                    │  │ └────────┘ │                     │
                                    └─────────────────────────────────────┘
```

### 检索流程（三路融合）

```
用户查询
    │
    ├──► Route 1: 向量检索 (Chroma ANN)
    │     文本 → DashScope embedding → 余弦相似度 → top_k
    │
    ├──► Route 2: Parent-Child 补充检索
    │     用子块检索 → 映射到父菜谱 → 补充评分
    │
    ├──► Route 3: BM25 关键词检索
    │     jieba 分词 → BM25 打分 → top_k
    │
    └──► RRF 融合排序
           score = 1/(k+rank_v) + 1/(k+rank_b25)
           → 重排序 → top_n → LLM 生成
```

## 快速开始

### 前置条件

- Python 3.10+
- Node.js 18+
- DashScope API Key（阿里云百炼平台）
- Moonshot API Key（Kimi 开放平台）

### 1. 后端安装与启动

```bash
# 克隆仓库
git clone https://github.com/lyt999741852/pythonProject.git
cd pythonProject/backend

# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key:
#   DASHSCOPE_API_KEY=sk-xxx
#   MOONSHOT_API_KEY=sk-xxx

# 启动后端服务
python -m app.api.server
```

服务默认运行在 `http://localhost:8001`。

### 2. 前端安装与启动

```bash
cd pythonProject/frontend
npm install
npm run dev
```

前端默认运行在 `http://localhost:5173`。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | SSE 流式对话 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 菜谱统计 |
| GET | `/api/routes` | 路由列表（调试） |

### 对话示例

```bash
curl -N -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "糖醋鲤鱼怎么做？"}'

# SSE 响应：
# : connected
# data: {"token": "..."}
# ...
# data: [DONE]
```

## 项目结构

```
pythonProject/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── server.py          # FastAPI 服务 + SSE 端点
│   │   ├── core/rag/
│   │   │   ├── config.py          # RAG 配置参数
│   │   │   ├── data_processor.py  # 文本分块
│   │   │   ├── embedder.py        # Embedding 工厂
│   │   │   ├── engine.py          # 核心编排引擎
│   │   │   ├── keyword_search.py  # BM25 关键词检索
│   │   │   ├── llm.py             # LLM 工厂
│   │   │   ├── prompt_manager.py  # 双模式提示词管理
│   │   │   ├── ranker.py          # RRF 融合排序
│   │   │   └── vector_store.py    # Chroma 向量存储
│   │   ├── models/recipe.py       # Recipe 数据模型
│   │   ├── services/rag_service.py # RAGService 门面
│   │   ├── config.py              # 全局配置 (Settings)
│   │   └── recipe_parser.py       # .md 菜谱解析器
│   ├── dishes/                    # 菜谱源文件（300+ 道）
│   │   ├── aquatic/               # 水产
│   │   ├── breakfast/             # 早餐
│   │   ├── condiment/             # 酱料
│   │   ├── dessert/               # 甜品
│   │   ├── drink/                 # 饮品
│   │   ├── meat_dish/             # 肉类
│   │   ├── semi-finished/         # 半成品
│   │   ├── soup/                  # 汤羹
│   │   ├── staple/                # 主食
│   │   └── vegetable_dish/        # 素菜
│   ├── eval/                      # 评测模块
│   │   ├── dataset.py             # 数据集加载
│   │   ├── metrics.py             # 检索指标（Recall/MRR/NDCG）
│   │   ├── build_dataset.py       # 评测数据集构建
│   │   ├── run_recall_eval.py     # 召回率评测
│   │   ├── run_ablation.py        # 消融实验
│   │   └── run_gen_eval.py        # 生成质量评测
│   ├── tests/                     # 单元测试
│   ├── pyproject.toml
│   └── .env.example
├── frontend/
│   ├── src/                       # React 源码
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── package.json
├── docs/                          # 项目文档
│   ├── tech-stack-summary.md
│   └── rag-evaluation-plan.md
└── README.md
```

## 评测结果

### 召回率评测 (text-embedding-v4)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.9708 |
| Recall@3 | 1.0000 |
| Precision@1 | 0.9708 |
| MRR | 0.9831 |
| NDCG@3 | 0.9930 |

### 消融实验 (Ablation)

不同 top_k 和 RRF k 值组合的召回率对比，详见 `docs/rag-evaluation-plan.md`。

## 许可证

MIT
