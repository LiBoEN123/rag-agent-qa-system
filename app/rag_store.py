import json

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import Settings
from app.rag_ingest import collect_supported_files
from app.rag_retrieval import tokenize


INDEX_STORE_DIR = Settings.INDEX_STORE_DIR

MANIFEST_PATH = INDEX_STORE_DIR / "index_manifest.json"
CHUNKS_CACHE_PATH = INDEX_STORE_DIR / "chunks.json"
TEXTS_CACHE_PATH = INDEX_STORE_DIR / "texts.json"
DOCS_REGISTRY_CACHE_PATH = INDEX_STORE_DIR / "docs_registry.json"
EMBEDDINGS_CACHE_PATH = INDEX_STORE_DIR / "doc_embeddings.npy"


def build_index(all_chunks: list[dict], model=None, rerank_model=None):
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
        model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME)

    if rerank_model is None:
        rerank_model = CrossEncoder(Settings.RERANK_MODEL_NAME)

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
    if not MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_manifest(manifest: dict):
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

    return False, "目录和参数都没有变化，无需重建"


def index_cache_exists() -> bool:
    return all([
        CHUNKS_CACHE_PATH.exists(),
        TEXTS_CACHE_PATH.exists(),
        DOCS_REGISTRY_CACHE_PATH.exists(),
        EMBEDDINGS_CACHE_PATH.exists(),
    ])


def save_index_cache(
    chunks: list[dict],
    texts: list[str],
    docs_registry: list[dict],
    doc_embeddings
):
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


def load_index_cache(model=None, rerank_model=None):
    if not index_cache_exists():
        return {
            "ok": False,
            "message": "未找到本地索引缓存",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }

    try:
        chunks = json.loads(CHUNKS_CACHE_PATH.read_text(encoding="utf-8"))
        texts = json.loads(TEXTS_CACHE_PATH.read_text(encoding="utf-8"))
        docs_registry = json.loads(DOCS_REGISTRY_CACHE_PATH.read_text(encoding="utf-8"))
        doc_embeddings = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=False)

        tokenized_chunks = [tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_chunks) if tokenized_chunks else None

        if model is None:
            model = SentenceTransformer(Settings.EMBEDDING_MODEL_NAME)

        if rerank_model is None:
            rerank_model = CrossEncoder(Settings.RERANK_MODEL_NAME)

        return {
            "ok": True,
            "message": "已从本地加载索引",
            "chunks": chunks,
            "texts": texts,
            "docs_registry": docs_registry,
            "doc_embeddings": doc_embeddings,
            "bm25": bm25,
            "model": model,
            "rerank_model": rerank_model
        }

    except Exception as e:
        return {
            "ok": False,
            "message": f"加载索引缓存失败：{e}",
            "chunks": [],
            "texts": [],
            "docs_registry": [],
            "doc_embeddings": None,
            "bm25": None,
            "model": model,
            "rerank_model": rerank_model
        }