"""应用配置模块。

集中读取环境变量、解析项目路径，并提供模型、索引和服务运行参数。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_RUNTIME_VALUES = {"legacy", "langgraph"}


def normalize_agent_runtime(value: str | None) -> str:
    """规整 Agent Runtime 配置值，空值默认使用旧运行时。"""
    normalized = (value or "").strip().lower()
    return normalized or "legacy"


def validate_agent_runtime(value: str | None) -> str:
    """校验并返回规整后的 Agent Runtime 配置值。"""
    normalized = normalize_agent_runtime(value)
    if normalized not in AGENT_RUNTIME_VALUES:
        allowed = ", ".join(sorted(AGENT_RUNTIME_VALUES))
        raise ValueError(
            f"AGENT_RUNTIME 配置无效: {value!r}，可选值为: {allowed}"
        )
    return normalized


def env_bool(name: str, default: str = "false") -> bool:
    """从环境变量读取布尔配置，支持常见真值字符串。"""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def project_path(path: Path) -> Path:
    """把相对路径解析为项目根目录下的绝对路径。"""
    return path if path.is_absolute() else BASE_DIR / path


def default_model_cache_dir() -> Path:
    """返回模型权重的本地缓存目录，优先尊重显式环境变量。"""
    if os.getenv("MODEL_CACHE_DIR"):
        return project_path(Path(os.getenv("MODEL_CACHE_DIR")).expanduser())
    if os.getenv("HF_HUB_CACHE"):
        return project_path(Path(os.getenv("HF_HUB_CACHE")).expanduser())
    if os.getenv("HF_HOME"):
        return project_path(Path(os.getenv("HF_HOME")).expanduser()) / "hub"
    if os.getenv("SENTENCE_TRANSFORMERS_HOME"):
        return project_path(Path(os.getenv("SENTENCE_TRANSFORMERS_HOME")).expanduser())
    return BASE_DIR / ".cache" / "huggingface" / "hub"


class Settings:
    RAG_FILE_PATH=Path(os.getenv("RAG_FILE_PATH", str(BASE_DIR / 'data')))
    INDEX_STORE_DIR = Path(os.getenv("INDEX_STORE_DIR", str(BASE_DIR / "data" / "manifest"))) # index manifest directory

    CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma")))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "mini_rag_chunks")

    TOP_K = int(os.getenv("TOP_K", "5"))
    DECOMPOSE_TOP_K = int(os.getenv("DECOMPOSE_TOP_K", "10"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 发散程度参数，越低模型回答越稳，越保守；越高模型越发散，越创造。

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME",
                                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME",
                                  "BAAI/bge-reranker-v2-m3")
    MODEL_CACHE_DIR = default_model_cache_dir()

    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 2500))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

    LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")

    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 2))
    LLM_RETRY_SLEEP = int(os.getenv("LLM_RETRY_SLEEP", 1))
    LLM_CLIENT_MODE = os.getenv("LLM_CLIENT_MODE", "langchain").strip().lower()

    USE_LLM_JUDGE = env_bool("USE_LLM_JUDGE", "false")
    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or LLM_API_KEY
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
    JUDGE_API_URL = os.getenv("JUDGE_API_URL", LLM_API_URL)
    JUDGE_TIMEOUT = int(os.getenv("JUDGE_TIMEOUT", os.getenv("LLM_TIMEOUT", "60")))
    JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "1"))
    JUDGE_RETRY_SLEEP = int(os.getenv("JUDGE_RETRY_SLEEP", os.getenv("LLM_RETRY_SLEEP", "1")))
    JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "2048"))

    ENABLE_LLM_ROUTER = env_bool("ENABLE_LLM_ROUTER", "false")
    ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or LLM_API_KEY
    ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME", MODEL_NAME)
    ROUTER_API_URL = os.getenv("ROUTER_API_URL", LLM_API_URL)
    ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "10"))
    ROUTER_MAX_RETRIES = int(os.getenv("ROUTER_MAX_RETRIES", "1"))
    ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "512"))
    ROUTER_TEMPERATURE = float(os.getenv("ROUTER_TEMPERATURE", "0.0"))

    AGENT_RUNTIME = normalize_agent_runtime(os.getenv("AGENT_RUNTIME", "legacy"))
    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))

    ENABLE_AGENT_TRACE_STORE = env_bool("ENABLE_AGENT_TRACE_STORE", "false")
    SQLITE_DB_PATH = project_path(
        Path(
            os.getenv(
                "SQLITE_DB_PATH",
                str(BASE_DIR / "data" / "app.sqlite"),
            )
        ).expanduser()
    )

    ENABLE_QUERY_REWRITE = env_bool("ENABLE_QUERY_REWRITE", "true")
    QUERY_REWRITE_API_KEY = os.getenv("QUERY_REWRITE_API_KEY") or LLM_API_KEY
    QUERY_REWRITE_MODEL_NAME = os.getenv("QUERY_REWRITE_MODEL_NAME", MODEL_NAME)
    QUERY_REWRITE_API_URL = os.getenv("QUERY_REWRITE_API_URL", LLM_API_URL)
    QUERY_REWRITE_TIMEOUT = int(os.getenv("QUERY_REWRITE_TIMEOUT", "10"))
    QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "256"))
    QUERY_REWRITE_TEMPERATURE = float(os.getenv("QUERY_REWRITE_TEMPERATURE", "0.0"))

    DECOMPOSE_API_KEY = os.getenv("DECOMPOSE_API_KEY")
    DECOMPOSE_MODEL_NAME = os.getenv("DECOMPOSE_MODEL_NAME", "qwen2.5:7b")
    DECOMPOSE_API_URL = os.getenv("DECOMPOSE_API_URL", "http://localhost:11434/v1/chat/completions")
    # Enables automatic query decomposition. The decomposition model still decides
    # whether a specific question needs sub-queries.
    ENABLE_QUERY_DECOMPOSE = env_bool("ENABLE_QUERY_DECOMPOSE", "true")
    DECOMPOSE_TIMEOUT = int(os.getenv("DECOMPOSE_TIMEOUT", "15"))
    DECOMPOSE_MAX_TOKENS = int(os.getenv("DECOMPOSE_MAX_TOKENS", "512"))

    VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.35"))

    AUTO_REBUILD_ON_STARTUP = env_bool("AUTO_REBUILD_ON_STARTUP", "false") #是否强制启动就重建

    @classmethod
    def validate(cls):
        """校验关键配置是否存在并可访问。"""
        cls.AGENT_RUNTIME = validate_agent_runtime(cls.AGENT_RUNTIME)
        if not cls.LLM_API_KEY:
            raise ValueError("缺少 LLM_API_KEY 或 DEEPSEEK_API_KEY，请检查 .env")
        if not cls.RAG_FILE_PATH.exists():
            raise ValueError(f"RAG_FILE_PATH 不存在: {cls.RAG_FILE_PATH}")

    @classmethod
    def ensure_model_cache_dir(cls) -> Path:
        """创建并返回 embedding/rerank 模型的本地缓存目录。"""
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODEL_CACHE_DIR
