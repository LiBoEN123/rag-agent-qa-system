import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

class Settings:
    RAG_FILE_PATH=Path(os.getenv("RAG_FILE_PATH", str(BASE_DIR / 'data')))
    INDEX_STORE_DIR = Path(os.getenv("INDEX_STORE_DIR", str(BASE_DIR / "data" / "manifest"))) # 索引缓存存哪里

    TOP_K = int(os.getenv("TOP_K", "3"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 发散程度参数，越低模型回答越稳，越保守；越高模型越发散，越创造。

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME",
                                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME",
                                  "BAAI/bge-reranker-v2-m3")

    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 2500))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")

    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 2))
    LLM_RETRY_SLEEP = int(os.getenv("LLM_RETRY_SLEEP", 1))

    DECOMPOSE_API_KEY = os.getenv("DECOMPOSE_API_KEY")
    DECOMPOSE_MODEL_NAME = os.getenv("DECOMPOSE_MODEL_NAME", "qwen2.5:7b")
    DECOMPOSE_API_URL = os.getenv("DECOMPOSE_API_URL", "http://localhost:11434/v1")

    MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.03"))
    VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.35"))

    AUTO_REBUILD_ON_STARTUP = env_bool("AUTO_REBUILD_ON_STARTUP", "false") #是否强制启动就重建

    @classmethod
    def validate(cls):
        if not cls.LLM_API_KEY:
            raise ValueError("缺少 LLM_API_KEY，请检查 .env")
        if not cls.RAG_FILE_PATH.exists():
            raise ValueError(f"RAG_FILE_PATH 不存在: {cls.RAG_FILE_PATH}")























