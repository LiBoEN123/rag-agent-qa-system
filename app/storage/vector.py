"""持久 Chroma 配置与 embedding factory；不持有应用 runtime。"""

import torch
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import AppSettings, ensure_model_cache_dir, get_settings


def create_vector_store(
    embeddings: Embeddings | None,
    settings: AppSettings | None = None,
    *,
    create_collection_if_not_exists: bool = True,
) -> Chroma:
    """创建由 langchain-chroma 管理读写的 v2 VectorStore。"""
    resolved = settings or get_settings()
    resolved.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=resolved.chroma_index_collection_name,
        persist_directory=str(resolved.chroma_persist_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        create_collection_if_not_exists=create_collection_if_not_exists,
    )


def load_embedding_model(settings: AppSettings | None = None) -> Embeddings:
    """从本地缓存加载独立 embedding 实例，也允许调用方直接注入 Embeddings。"""
    resolved = settings or get_settings()
    return HuggingFaceEmbeddings(
        model=resolved.embedding_model_name,
        cache_folder=str(ensure_model_cache_dir(resolved)),
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
