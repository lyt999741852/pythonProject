# Vibe Cooking — 技术栈总结

## 项目概述

全栈智能烹饪助手。后端基于 LangChain + FastAPI 构建 RAG 管线，从 300+ 中文菜谱 Markdown 中检索，经 Moonshot LLM 流式生成回答；前端 React + Vite + Tailwind 提供对话 UI。

---

## 一、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (:5173)                      │
│   React 18 + TypeScript + Tailwind CSS + Vite 6         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  App.tsx (Chat UI) → chat.ts (SSE client)       │  │
│   └──────────────┬───────────────────────────────────┘  │
│                  │ POST /api/chat (proxy → :8001)       │
├──────────────────┼─────────────────────────────────────┤
│   Vite Proxy     │                                     │
└──────────────────┼─────────────────────────────────────┘
                   │ SSE (text/event-stream)
┌──────────────────┼─────────────────────────────────────┐
│              Backend (:8001)                            │
│   FastAPI + uvicorn                                     │
│   ┌──────────────────────────────────────────────────┐  │
│   │  server.py (lifespan, CORS, 3 endpoints)         │  │
│   └──────────────┬───────────────────────────────────┘  │
│                  │ process_query()                      │
│   ┌──────────────┴───────────────────────────────────┐  │
│   │          RAGService (orchestrator)                │  │
│   │  ┌─────────────┐  ┌──────────┐  ┌─────────────┐  │  │
│   │  │ Ingestion   │  │Retriever │  │  Generator  │  │  │
│   │  │ Service     │  │(3-route) │  │(streaming)  │  │  │
│   │  └──────┬──────┘  └────┬─────┘  └──────┬──────┘  │  │
│   └─────────┼──────────────┼───────────────┼─────────┘  │
│             │              │               │            │
│   ┌─────────┴──────────────┴───────────────┴─────────┐  │
│   │           LangChain Components                    │  │
│   │  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │  │
│   │  │OpenAI    │  │ Chroma   │  │ ChatOpenAI     │  │  │
│   │  │Embeddings│  │(向量库)  │  │(Moonshot/Kimi) │  │  │
│   │  │(DashScope)│  │          │  │                │  │  │
│   │  └──────────┘  └──────────┘  └────────────────┘  │  │
│   │  ┌──────────────────────────────────────────────┐  │  │
│   │  │ BM25Retriever (jieba 分词 + rank_bm25)       │  │  │
│   │  └──────────────────────────────────────────────┘  │  │
│   └────────────────────────────────────────────────────┘  │
│                                                           │
│   ┌────────────────────────────────────────────────────┐  │
│   │  持久层                                              │  │
│   │  ├── data/parents.json   (322 条 parent recipe)     │  │
│   │  └── data/chroma/        (Chroma SQLite 向量库)     │  │
│   └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 二、后端技术栈

### 2.1 框架与运行

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | **FastAPI** 0.110+ | 3 个 REST 端点 + SSE 流式响应 |
| ASGI 服务器 | **uvicorn** 0.27+ | 异步 WSGI 服务 |
| 配置管理 | **pydantic-settings** | 从 `.env` 加载 14 项配置 |
| 数据验证 | **pydantic** 2.x | 6 个数据模型，输入输出校验 |

### 2.2 RAG 管线 (9 模块 + 编排)

| 模块 | 文件 | 技术 | 职责 |
|------|------|------|------|
| 配置 | `config.py` | dataclass | RAGConfig 参数容器 |
| 嵌入 | `embedder.py` | OpenAIEmbeddings | DashScope text-embedding-v3 |
| 向量库 | `vector_store.py` | Chroma | 持久化向量存储 (cosine) |
| 关键词 | `keyword_search.py` | BM25 + jieba | 中文分词 + 关键词检索 |
| 数据处理 | `data_processor.py` | 纯函数 | 构建全文 + 按步骤分块 |
| 排序融合 | `ranker.py` | RRF | 加权融合向量+BM25 结果 |
| 提示词 | `prompt_manager.py` | 模板字符串 | RAG/对话双模式提示词 |
| 异常 | `exceptions.py` | 异常层次 | 5 类自定义异常 |
| 编排 | `engine.py` | 4 个类 | 总装所有组件 |

**编排层 (engine.py) 4 个类：**

- **RecipeIngestionService**: Parent-Child 分块策略，parent 存内存 + JSON，child 存 Chroma
- **RecipeRetriever**: 三路检索 → RRF 融合
- **StreamingGenerator**: 异步流式生成，双模式提示词路由
- **RAGService**: 顶层门面，生命周期管理

### 2.3 检索流程

```
用户查询
    │
    ▼
┌──────────────┐
│  1. Embedding │ ── DashScope(text-embedding-v3) ──→ 失败时降级对话模式
└──────┬───────┘
       │ query_embedding
       ▼
┌───────────────────────────────────────────┐
│  2. 三路检索                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Route 1  │  │ Route 2  │  │ Route 3  │  │
│  │ 向量检索  │→│ 父文档富化 │  │ BM25    │  │
│  │ (Chroma) │  │(不参与RRF)│  │ (jieba)  │  │
│  └────┬─────┘  └──────────┘  └────┬─────┘  │
│       └──────────┬────────────────┘         │
│                  ▼                          │
│          ┌──────────────┐                   │
│          │ RRF 融合     │                   │
│          │ (k=60, top=3)│                   │
│          └──────────────┘                   │
└───────────────────────────────────────────┘
                    │ top-N results
                    ▼
┌───────────────────────────────────────────┐
│  3. 提示词选择                             │
│  有结果 → RAG_SYSTEM_PROMPT + context     │
│  无结果 → CHAT_SYSTEM_PROMPT (自由对话)    │
└──────────────────┬────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│  4. LLM 流式生成                           │
│  Moonshot (moonshot-v1-8k) via SSE       │
│  ChatOpenAI.astream()                    │
└───────────────────────────────────────────┘
```

### 2.4 外部服务

| 服务 | API | 模型 | 用途 | 状态 |
|------|-----|------|------|------|
| 阿里云 DashScope | OpenAI 兼容 | `text-embedding-v3` | 文本嵌入 | ❌ 免费额度耗尽 |
| Moonshot (月之暗面) | OpenAI 兼容 | `moonshot-v1-8k` | LLM 生成 | ✅ 正常运行 |

### 2.5 持久化

- **parents.json**: 322 条父文档（菜谱完整文本），每次启动加载，无需重新解析
- **Chroma SQLite**: 子块向量索引（当前为空 — DashScope 额度问题）

---

## 三、前端技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| UI 框架 | **React 18** | 组件化 SPA |
| 构建工具 | **Vite 6** | 极速 HMR 开发服务 |
| 样式 | **Tailwind CSS 3** | 原子化 CSS，橙色主题色 |
| 语言 | **TypeScript** | 类型安全 |
| 流式通信 | **Fetch API + SSE** | 原生日志流解析 |

无额外 UI 库依赖 — AbortController 管理请求取消，AsyncGenerator 处理流式 token。

---

## 四、开发工具

| 工具 | 用途 |
|------|------|
| Python 3.12 | 运行时 |
| Conda (all-in-rag) | 虚拟环境 |
| pytest 9.0 | 33 个单元测试 |
| Node.js | 前端构建 |

---

## 五、数据流

```
.md 文件 (dishes/)                 运行时交互
    │                                    │
    ▼                                    ▼
recipe_parser.py ←── 322 个 Recipe ───→ FastAPI server
    │                                    │
    ▼                                    ▼
IngestionService                POST /api/chat {query}
    │                                    │
    ├─→ parents.json (persist)           ▼
    └─→ Chroma (child chunks)     RAGService.process_query()
                                         │
                                         ▼
                                  StreamingGenerator
                                    ├─ embed query
                                    ├─ 3-way retrieve
                                    ├─ RRF fuse
                                    └─ LLM astream
                                         │
                                         ▼
                                  SSE: token → token → [DONE]
                                         │
                                         ▼
                                  React UI (逐 token 渲染)
```
