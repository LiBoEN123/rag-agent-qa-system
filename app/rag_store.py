import json

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.chroma_store import (
    ChromaUnavailableError,
    chroma_collection_has_chunks,
    is_chroma_available,
    load_chunks_from_chroma,
    persist_chunks_to_chroma,
)
from app.config import Settings
from app.rag_ingest import collect_supported_files
from app.rag_retrieval import tokenize
from app.untils import get_logger


devcie = 'cuda' if torch.cuda.is_available() else "cpu"
logger = get_logger('chat_service')

INDEX_STORE_DIR = Settings.INDEX_STORE_DIR

MANIFEST_PATH = INDEX_STORE_DIR / "index_manifest.json"
CHUNKS_CACHE_PATH = INDEX_STORE_DIR / "chunks.json"
TEXTS_CACHE_PATH = INDEX_STORE_DIR / "texts.json"
DOCS_REGISTRY_CACHE_PATH = INDEX_STORE_DIR / "docs_registry.json"
EMBEDDINGS_CACHE_PATH = INDEX_STORE_DIR / "doc_embeddings.npy"


def build_index(all_chunks: list[dict], model=None, rerank_model=None):
    """基于 chunk 文本构建 BM25、向量嵌入和重排模型。"""
    if not all_chunks:
        return {
            "texts": [],
            "bm25": None,
            "doc_embeddings": None,
            "model": model,
            "rerank_model": rerank_model
        }

    local_texts = [chunk["text"] for chunk in all_chunks]
    tokenized_texts = [tokenize(text) for text in local_texts]

    if model is None:
        model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME, device=devcie)

    if rerank_model is None:
        rerank_model = CrossEncoder(Settings.RERANK_MODEL_NAME, device=devcie)

    local_doc_embeddings = model.encode(
        local_texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    local_bm25 = BM25Okapi(tokenized_texts)

    return {
        "texts": local_texts,
        "bm25": local_bm25,
        "doc_embeddings": local_doc_embeddings,
        "model": model,
        "rerank_model": rerank_model
    }


def build_file_snapshot(data_dir: str):
    """记录知识库文件的路径、大小和修改时间快照。"""
    files = collect_supported_files(data_dir)
    snapshot = []

    for file_path in files:
        stat = file_path.stat()
        snapshot.append({
            "path": str(file_path.resolve()),
            "size": stat.st_size,
            "time": stat.st_mtime
        })

    snapshot.sort(key=lambda x: x["path"])
    return snapshot


def build_current_manifest(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """生成当前索引构建配置和文件快照的 manifest。"""
    return {
        "pipeline_version": 1,
        "embedding_model": Settings.EMBEDDING_MODEL_NAME,
        "rerank_model": Settings.RERANK_MODEL_NAME,
        "build_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "files": build_file_snapshot(data_dir),
    }


def load_manifest():
    """从磁盘读取历史索引 manifest。"""
    if not MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict):
    """将当前 manifest 写入磁盘。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def needs_rebuild(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
) -> tuple[bool, str]:
    """对比历史 manifest 与当前状态判断是否需要重建索引。"""
    old_manifest = load_manifest()
    current_manifest = build_current_manifest(data_dir, chunk_size, chunk_overlap)

    if old_manifest is None:
        return True, "没有找到历史 manifest，需要首次构建"

    if old_manifest.get("pipeline_version") != current_manifest.get("pipeline_version"):
        return True, "pipeline_version 变化，需要重建"

    if old_manifest.get("embedding_model") != current_manifest.get("embedding_model"):
        return True, "embedding 模型变化，需要重建"

    if old_manifest.get("rerank_model") != current_manifest.get("rerank_model"):
        return True, "rerank 模型变化，需要重建"

    if old_manifest.get("build_config") != current_manifest.get("build_config"):
        return True, "chunk 参数变化，需要重建"

    if old_manifest.get("files") != current_manifest.get("files"):
        return True, "data 目录文件变化，需要重建"

    if Settings.ENABLE_CHROMA and is_chroma_available() and not chroma_collection_has_chunks():
        return True, "ChromaDB collection 中未找到 chunk，需要重建"

    return False, "目录和参数都没有变化，无需重建"


def legacy_index_cache_exists() -> bool:
    """检查旧版 JSON/NPY 索引缓存文件是否齐全。"""
    return all([
        CHUNKS_CACHE_PATH.exists(),
        TEXTS_CACHE_PATH.exists(),
        DOCS_REGISTRY_CACHE_PATH.exists(),
        EMBEDDINGS_CACHE_PATH.exists(),
    ])


def index_cache_exists() -> bool:
    """检查当前可用索引缓存是否存在，优先识别 ChromaDB。"""
    if DOCS_REGISTRY_CACHE_PATH.exists() and chroma_collection_has_chunks():
        return True
    return legacy_index_cache_exists()




def get_index_cache_source() -> str:
    """Return the currently usable persisted index source."""
    if DOCS_REGISTRY_CACHE_PATH.exists() and chroma_collection_has_chunks():
        return "chroma"
    if legacy_index_cache_exists():
        return "legacy"
    return "none"

def save_index_cache(
    chunks: list[dict],
    texts: list[str],
    docs_registry: list[dict],
    doc_embeddings
):
    """将 chunks、文本、文档注册表和向量保存到 ChromaDB，并保留旧缓存兜底。"""
    INDEX_STORE_DIR.mkdir(parents=True, exist_ok=True)

    CHUNKS_CACHE_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    TEXTS_CACHE_PATH.write_text(
        json.dumps(texts, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    DOCS_REGISTRY_CACHE_PATH.write_text(
        json.dumps(docs_registry, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    np.save(EMBEDDINGS_CACHE_PATH, doc_embeddings)

    if not Settings.ENABLE_CHROMA:
        logger.info("[index] ChromaDB disabled, saved legacy index cache only")
        return

    try:
        saved_count = persist_chunks_to_chroma(
            chunks=chunks,
            embeddings=doc_embeddings,
            reset=True,
        )
        logger.info(f"[index] saved chunks to ChromaDB count={saved_count}")
    except ChromaUnavailableError as e:
        logger.warning(f"[index] ChromaDB unavailable, legacy cache saved only: {e}")
    except Exception:
        logger.exception("[index] failed to save chunks to ChromaDB")
        raise


def _load_models(model=None, rerank_model=None):
    """Load embedding and rerank models when cache data is available."""
    if model is None:
        model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME, device=devcie)

    if rerank_model is None:
        rerank_model = CrossEncoder(Settings.RERANK_MODEL_NAME, device=devcie)

    return model, rerank_model


def _build_bm25(texts: list[str]):
    tokenized_chunks = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_chunks) if tokenized_chunks else None


def _load_index_cache_from_chroma(model=None, rerank_model=None):
    """从 ChromaDB 恢复 chunk 和 BM25，向量检索阶段继续查询 ChromaDB。"""
    if not (DOCS_REGISTRY_CACHE_PATH.exists() and chroma_collection_has_chunks()):
        return None

    chunks = load_chunks_from_chroma()
    texts = [chunk["text"] for chunk in chunks]
    docs_registry = json.loads(DOCS_REGISTRY_CACHE_PATH.read_text(encoding="utf-8"))
    bm25 = _build_bm25(texts)
    model, rerank_model = _load_models(model=model, rerank_model=rerank_model)

    return {
        "ok": True,
        "message": "已从 ChromaDB 加载索引",
        "source": "chroma",
        "chunks": chunks,
        "texts": texts,
        "docs_registry": docs_registry,
        "doc_embeddings": None,
        "bm25": bm25,
        "model": model,
        "rerank_model": rerank_model
    }


def _load_legacy_index_cache(model=None, rerank_model=None):
    """从旧版 JSON/NPY 缓存恢复索引数据、BM25 和模型实例。"""
    if not legacy_index_cache_exists():
        return {
            "ok": False,
            "message": "未找到本地索引缓存",
            "source": "none",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }

    chunks = json.loads(CHUNKS_CACHE_PATH.read_text(encoding="utf-8"))
    texts = json.loads(TEXTS_CACHE_PATH.read_text(encoding="utf-8"))
    docs_registry = json.loads(DOCS_REGISTRY_CACHE_PATH.read_text(encoding="utf-8"))
    doc_embeddings = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=False)
    bm25 = _build_bm25(texts)
    model, rerank_model = _load_models(model=model, rerank_model=rerank_model)

    return {
        "ok": True,
        "message": "已从本地 legacy 缓存加载索引",
        "source": "legacy",
        "chunks": chunks,
        "texts": texts,
        "docs_registry": docs_registry,
        "doc_embeddings": doc_embeddings,
        "bm25": bm25,
        "model": model,
        "rerank_model": rerank_model
    }


def load_index_cache(model=None, rerank_model=None):
    """从缓存恢复索引数据、BM25 和模型实例，优先使用 ChromaDB。"""
    try:
        chroma_result = _load_index_cache_from_chroma(model=model, rerank_model=rerank_model)
        if chroma_result is not None:
            return chroma_result
    except ChromaUnavailableError as e:
        logger.warning(f"[index] ChromaDB unavailable while loading cache: {e}")
    except Exception as e:
        logger.warning(f"[index] 加载 ChromaDB 索引失败，尝试 legacy 缓存：{e}")

    try:
        return _load_legacy_index_cache(model=model, rerank_model=rerank_model)
    except Exception as e:
        return {
            "ok": False,
            "message": f"加载索引缓存失败：{e}",
            "source": "none",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }