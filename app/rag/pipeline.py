"""RAG 索引与检索门面模块。

维护内存索引状态，并组织索引生命周期、检索、重排和调试能力。
"""

from typing import Any, Dict, List

from app.config import get_settings
from app.rag.debug import (
    get_chunk_full_text as get_chunk_full_text_impl,
    preview_chunks as preview_chunks_impl,
    preview_docs as preview_docs_impl,
)
from app.rag.indexing import load_local_langchain_documents, sync_documents
from app.rag.retrieval import (
    retrieve as retrieve_impl,
    rerank as rerank_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
    retrieve_with_sub_queries,
)
from app.rag.store import (
    build_current_manifest,
    _load_embedding_model,
    index_cache_exists as index_cache_exists_impl,
    load_index_cache as load_index_cache_impl,
    needs_rebuild as needs_rebuild_impl,
    save_manifest,
)
# 进程内索引状态。
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
def _publish_runtime_bundle(bundle: dict) -> None:
    """只在 index、runtime rebuild 与验证全部成功后替换进程状态。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    chunks = bundle["chunks"]
    texts = bundle["texts"]
    docs_registry = bundle["docs_registry"]
    doc_embeddings = bundle["doc_embeddings"]
    bm25 = bundle["bm25"]
    model = bundle["model"]
    rerank_model = bundle["rerank_model"]
    RAG_READY = bool(chunks)
    RAG_INIT_MESSAGE = "OK" if chunks else NO_DOCUMENTS_MESSAGE
    INDEX_SOURCE = bundle.get("source", "none") if chunks else "none"


def _sync_and_build_runtime(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[dict, dict[str, int]]:
    """增量同步 Local corpus，并在发布前构建完整 runtime bundle。"""
    settings = get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    documents = load_local_langchain_documents(
        data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding_model = (
        _load_embedding_model()
        if force_rebuild or model is None
        else model
    )
    sync_result = sync_documents(
        documents,
        embedding_model,
        settings=settings,
        force_rebuild=force_rebuild,
    )

    if not documents:
        return (
            {
                "ok": True,
                "source": "none",
                "chunks": [],
                "texts": [],
                "docs_registry": [],
                "doc_embeddings": None,
                "bm25": None,
                "model": embedding_model,
                "rerank_model": rerank_model,
            },
            sync_result,
        )

    bundle = load_index_cache_impl(
        model=embedding_model,
        rerank_model=rerank_model,
    )
    if not bundle["ok"]:
        raise RuntimeError(bundle["message"])
    return bundle, sync_result


def rebuild_index(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """通过 LangChain 增量同步并发布内存 RAG runtime。"""
    bundle, sync_result = _sync_and_build_runtime(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    _publish_runtime_bundle(bundle)
    return sync_result

def rebuild_and_persist_index(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[bool, str]:
    """增量同步 v2 索引，验证 runtime 后保存 manifest 并原子发布内存引用。"""
    settings = get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    try:
        bundle, sync_result = _sync_and_build_runtime(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_rebuild=force_rebuild,
        )
        save_manifest(build_current_manifest(data_dir, chunk_size, chunk_overlap))
    except Exception as exc:
        return False, f"LangChain 增量索引失败：{exc}"

    _publish_runtime_bundle(bundle)
    summary = ", ".join(f"{key}={value}" for key, value in sync_result.items())
    if not bundle["chunks"]:
        return True, f"Local corpus 已同步为空（{summary}）"
    return True, f"LangChain 增量索引已同步并发布（{summary}）"


# =========================
# 缓存与清单
# =========================
def needs_rebuild(
    data_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """代理底层 manifest 检查，判断当前知识库是否需要重建。"""
    settings = get_settings()
    chunk_size = settings.chunk_size if chunk_size is None else chunk_size
    chunk_overlap = (
        settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    )
    return needs_rebuild_impl(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def index_cache_exists():
    """检查本地索引缓存文件是否齐全。"""
    return index_cache_exists_impl()


def save_index_cache():
    """拒绝旧 direct-write API；索引写入必须经过 LangChain sync。"""
    raise RuntimeError(
        "save_index_cache 已停用，请使用 rebuild_and_persist_index"
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

    if not result["chunks"]:
        RAG_READY = False
        RAG_INIT_MESSAGE = "索引缓存为空"
        INDEX_SOURCE = result.get("source", "none")
        return False, RAG_INIT_MESSAGE

    _publish_runtime_bundle(result)
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
