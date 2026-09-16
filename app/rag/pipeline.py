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
from app.rag.indexing import (
    EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE,
    LOCAL_RECORD_NAMESPACE,
    candidate_to_langchain_documents,
    create_record_manager,
    create_vector_store,
    load_local_langchain_documents,
    sync_documents,
)
from app.rag.policy import should_refuse
from app.rag.retrieval import (
    retrieve as retrieve_impl,
    rerank as rerank_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
    retrieve_with_sub_queries,
)
from app.rag.scoring import safe_float
from app.rag.store import (
    build_current_manifest,
    _load_embedding_model,
    load_index_cache as load_index_cache_impl,
    save_manifest,
)
from app.storage.ingestion import SQLiteCandidateRepository

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
    """普通同步只更新 Local；force 从完整 authoritative corpus 重建共享索引。"""
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
    documents_by_namespace = {LOCAL_RECORD_NAMESPACE: documents}
    if force_rebuild:
        documents_by_namespace.update(
            (namespace, [])
            for namespace in EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.values()
        )
        candidates = SQLiteCandidateRepository(
            settings.sqlite_db_path
        ).load_latest_successfully_indexed()
        for candidate in candidates:
            namespace = EXTERNAL_RECORD_NAMESPACE_BY_SOURCE_TYPE.get(
                candidate.document.source_type
            )
            if namespace is None:
                raise ValueError("成功 index candidate 的 external source type 不支持恢复")
            documents_by_namespace[namespace].extend(
                candidate_to_langchain_documents(
                    candidate,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

    embedding_model = (
        _load_embedding_model()
        if force_rebuild or model is None
        else model
    )
    if force_rebuild:
        # GLOBAL_RESET：全部 candidate 已验证、切分后，才能清空共享 v2 collection。
        create_vector_store(embedding_model, settings).delete_collection()
        for namespace in documents_by_namespace:
            manager = create_record_manager(settings, namespace=namespace)
            keys = manager.list_keys()
            if keys:
                manager.delete_keys(keys)

    sync_result: dict[str, int] = {}
    for namespace, namespace_documents in documents_by_namespace.items():
        result = sync_documents(
            namespace_documents,
            embedding_model,
            settings=settings,
            namespace=namespace,
        )
        for key, count in result.items():
            sync_result[key] = sync_result.get(key, 0) + count

    bundle = load_index_cache_impl(
        model=embedding_model,
        rerank_model=rerank_model,
    )
    if not bundle["ok"]:
        # Cache loader 将空 collection 视为无缓存；严格读取区分真正空库与读取失败。
        store = create_vector_store(
            None, settings, create_collection_if_not_exists=False
        )
        if store.get(limit=1, include=[])["ids"]:
            raise RuntimeError(bundle["message"])
        bundle.update(ok=True, message=NO_DOCUMENTS_MESSAGE)
    return bundle, sync_result


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
        return True, f"共享知识库已同步为空（{summary}）"
    return True, f"LangChain 增量索引已同步并发布（{summary}）"


# =========================
# 持久化索引恢复
# =========================
def load_index_cache(*, allow_empty: bool = False) -> tuple[bool, str]:
    """恢复全局 RAG 状态；来源操作可在严格确认空库后发布空 runtime。"""
    global chunks, texts, docs_registry, bm25, model, rerank_model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE, INDEX_SOURCE

    result = load_index_cache_impl(model=model)

    if allow_empty and (not result["ok"] or not result["chunks"]):
        try:
            store = create_vector_store(None, create_collection_if_not_exists=False)
            if store.get(limit=1, include=[])["ids"]:
                return False, result["message"]
        except Exception:
            return False, "无法验证 ChromaDB 空索引"
        result.update(ok=True, message=NO_DOCUMENTS_MESSAGE)
        _publish_runtime_bundle(result)
        return True, result["message"]

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
def run_diagnostic_search(question: str, top_k: int) -> dict[str, Any]:
    """按当前索引执行单查询检索、重排和证据门控，不进入问答工作流。"""
    rag_status = get_rag_status()
    if not rag_status["ready"]:
        return {
            "status": "empty_index",
            "message": rag_status["message"],
            "reason": "知识库未准备完成",
            "chunks": [],
        }

    retrieved = search_relevant_chunks(question, top_k)
    if not retrieved:
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": "未检索到结果",
            "chunks": [],
        }

    final_chunks = rerank(retrieved, question)[:top_k]
    if not final_chunks:
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": "重排后没有可用结果",
            "chunks": [],
        }

    if should_refuse(final_chunks):
        max_vector_score = max(
            safe_float(item.get("vector_score")) for item in final_chunks
        )
        return {
            "status": "no_evidence",
            "message": "success",
            "reason": f"证据分数低于阈值：max_vector={max_vector_score:.4f}",
            "chunks": [],
        }

    return {
        "status": "ok",
        "message": "success",
        "reason": "检索成功",
        "chunks": final_chunks,
    }


def preview_docs():
    """返回当前知识库文档注册表的调试预览。"""
    return preview_docs_impl(docs_registry)


def preview_chunks(max_len:int, limit: int = 70, file_name: str | None = None):
    """返回当前知识库 chunk 的调试预览。"""
    return preview_chunks_impl(chunks, max_len=max_len, limit=limit, file_name=file_name)


def get_chunk_full_text(chunk_id: int | str):
    """根据 chunk_id 返回对应 chunk 的完整文本。"""
    return get_chunk_full_text_impl(chunks, chunk_id)
