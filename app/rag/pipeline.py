"""RAG 索引与检索门面模块。

维护内存索引状态，并组织索引生命周期、检索、重排和调试能力。
"""

from typing import Any, Dict, List

from app.config import Settings
from app.rag.debug import (
    get_chunk_full_text as get_chunk_full_text_impl,
    preview_chunks as preview_chunks_impl,
    preview_docs as preview_docs_impl,
)
from app.rag.ingest import clean_documents, load_files, split_chunks
from app.rag.retrieval import (
    retrieve as retrieve_impl,
    rerank as rerank_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
    retrieve_with_sub_queries,
)
from app.rag.store import (
    build_current_manifest,
    build_index,
    get_index_cache_source as get_index_cache_source_impl,
    index_cache_exists as index_cache_exists_impl,
    load_index_cache as load_index_cache_impl,
    needs_rebuild as needs_rebuild_impl,
    save_index_cache as save_index_cache_impl,
    save_manifest,
)
# In-process index state.
chunks: list[dict] = []
texts: list[str] = []
docs_registry: list[dict] = []

bm25 = None
model = None
rerank_model = None
doc_embeddings = None

RAG_READY = False
RAG_INIT_MESSAGE = "RAG 尚未初始化"
INDEX_SOURCE = "none"
NO_DOCUMENTS_MESSAGE = "当前知识库为空，请先在 data 目录中放入文档后再提问"
STOP_WORD = "我无法根据提供的文档确定答案"


# =========================
# 状态函数
# =========================
def get_rag_status():
    """返回当前内存索引是否可用，以及文档和 chunk 数量。"""
    return {
        "ready": RAG_READY,
        "message": RAG_INIT_MESSAGE,
        "doc_count": len(docs_registry),
        "chunk_count": len(chunks),
        "index_source": INDEX_SOURCE
    }


# =========================
# 初始化 / 构建
# =========================
def rebuild_index(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """从 data 目录重新加载文档、切分 chunk，并重建内存索引。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    chunks = []
    texts = []
    docs_registry = []
    bm25 = None
    doc_embeddings = None
    model = None
    rerank_model = None
    RAG_READY = False
    RAG_INIT_MESSAGE = "RAG 尚未初始化"
    INDEX_SOURCE = "none"

    raw_docs, docs_registry_local = load_files(data_dir)
    if not raw_docs:
        docs_registry = []
        RAG_INIT_MESSAGE = NO_DOCUMENTS_MESSAGE
        return

    cleaned_docs = clean_documents(raw_docs)

    all_chunks, chunk_count_map = split_chunks(
        cleaned_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in docs_registry_local:
        doc["chunk_count"] = chunk_count_map.get(doc["source_path"], 0)

    index_bundle = build_index(all_chunks, model=model, rerank_model=rerank_model)

    chunks = all_chunks
    texts = index_bundle["texts"]
    bm25 = index_bundle["bm25"]
    doc_embeddings = index_bundle["doc_embeddings"]
    docs_registry = docs_registry_local
    model = index_bundle["model"]
    rerank_model = index_bundle["rerank_model"]

    if not chunks:
        RAG_INIT_MESSAGE = "文档已加载，但切分后没有可用 chunk"
        return

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"
    INDEX_SOURCE = "memory"

def rebuild_and_persist_index(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
) -> tuple[bool, str]:
    """重建内存索引并在成功后保存缓存和 manifest。"""
    global INDEX_SOURCE
    rebuild_index(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not RAG_READY:
        return False, RAG_INIT_MESSAGE

    save_index_cache()
    save_manifest(build_current_manifest(data_dir, chunk_size, chunk_overlap))
    INDEX_SOURCE = get_index_cache_source_impl()

    return True, "索引已重建，并已保存到磁盘"


# =========================
# cache / manifest
# =========================
def needs_rebuild(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """代理底层 manifest 检查，判断当前知识库是否需要重建。"""
    return needs_rebuild_impl(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def index_cache_exists():
    """检查本地索引缓存文件是否齐全。"""
    return index_cache_exists_impl()


def save_index_cache():
    """将当前内存中的索引数据保存到磁盘缓存。"""
    save_index_cache_impl(
        chunks=chunks,
        doc_embeddings=doc_embeddings
    )


def load_index_cache() -> tuple[bool, str]:
    """从磁盘缓存加载索引并恢复全局 RAG 状态。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    result = load_index_cache_impl(model=model)

    if not result["ok"]:
        RAG_READY = False
        RAG_INIT_MESSAGE = result["message"]
        INDEX_SOURCE = result.get("source", "none")
        return False, RAG_INIT_MESSAGE

    chunks = result["chunks"]
    texts = result["texts"]
    docs_registry = result["docs_registry"]
    doc_embeddings = result["doc_embeddings"]
    bm25 = result["bm25"]
    model = result["model"]
    rerank_model = result["rerank_model"]

    if not chunks:
        RAG_READY = False
        RAG_INIT_MESSAGE = "索引缓存为空"
        INDEX_SOURCE = result.get("source", "none")
        return False, RAG_INIT_MESSAGE

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"
    INDEX_SOURCE = result.get("source", get_index_cache_source_impl())
    return True, result["message"]


# =========================
# 查询阶段：检索包装
# =========================
def retrieve(question: str, top_k: int = 3):
    """使用当前全局索引执行一次基础召回。"""
    return retrieve_impl(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k
    )

def rerank(candidates: list[dict], question: str=""):
    """使用当前全局重排模型对候选 chunk 重新排序。"""
    return rerank_impl(
        rerank_model=rerank_model,
        candidates=candidates,
        question=question,
    )


def retrieve_with_sub_queries_from_current_index(
    question: str,
    sub_queries: list[dict],
    final_top_k: int,
) -> list[dict]:
    """使用当前进程索引执行现有多查询检索，不复制索引或检索逻辑。"""
    return retrieve_with_sub_queries(
        question=question,
        sub_queries=sub_queries,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        rerank_model=rerank_model,
        final_top_k=final_top_k,
    )


def search_relevant_chunks(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """使用当前全局索引检索与问题相关的 chunk。"""
    return search_relevant_chunks_impl(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k
    )


# =========================
# 调试接口包装
# =========================
def preview_docs():
    """返回当前知识库文档注册表的调试预览。"""
    return preview_docs_impl(docs_registry)


def preview_chunks(max_len:int, limit: int = 70, file_name: str | None = None):
    """返回当前知识库 chunk 的调试预览。"""
    return preview_chunks_impl(chunks, max_len=max_len, limit=limit, file_name=file_name)


def get_chunk_full_text(chunk_id: int | str):
    """根据 chunk_id 返回对应 chunk 的完整文本。"""
    return get_chunk_full_text_impl(chunks, chunk_id)
