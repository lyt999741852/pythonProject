from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration for the RAG service.

    Loads from .env file automatically via pydantic-settings.
    """

    # ---- DashScope (Alibaba Cloud Embedding) ----
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    EMBEDDING_DIMENSIONS: int = 1024

    # ---- Moonshot (Kimi LLM) ----
    MOONSHOT_API_KEY: str = ""
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"
    LLM_MODEL: str = "moonshot-v1-8k"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # ---- Chroma Persistence ----
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"

    # ---- Retrieval Parameters ----
    VECTOR_SEARCH_TOP_K: int = 5
    BM25_SEARCH_TOP_K: int = 5
    RRF_K_CONSTANT: int = 60
    FINAL_TOP_N: int = 3

    # ---- Parent Persistence ----
    PARENTS_DATA_PATH: str = "./data/parents.json"

    # ---- Retry Configuration ----
    EMBEDDING_MAX_RETRIES: int = 3
    EMBEDDING_RETRY_BASE_DELAY: float = 2.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
