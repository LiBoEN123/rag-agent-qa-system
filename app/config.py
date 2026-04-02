import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    RAG_FILE_PATH=Path(os.getenv("RAG_FILE_PATH", str(BASE_DIR / 'data')))
    TOP_K = int(os.getenv("TOP_K", "3"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 发散程度参数，越低模型回答越稳，越保守；越高模型越发散，越创造。

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME",
                                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")

    HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.40"))
    VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.35"))

    @classmethod
    def validate(cls):
        if not cls.LLM_API_KEY:
            raise ValueError("缺少 LLM_API_KEY，请检查 .env")
        if not cls.RAG_FILE_PATH.exists():
            raise ValueError(f"RAG_FILE_PATH 不存在: {cls.RAG_FILE_PATH}")























