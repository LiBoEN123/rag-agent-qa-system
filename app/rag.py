from typing import Any, Dict, List
import time

from app.untils import get_logger
from app.config import Settings
from app.prompts import NO_ANSWER_FALLBACK
from app.rag_debug import (
    get_chunk_full_text as get_chunk_full_text_impl,
    preview_chunks as preview_chunks_impl,
    preview_docs as preview_docs_impl,
)
from app.rag_generation import (
    build_context,
    call_llm,
    estimate_confidence,
    format_reference,
    parse_llm_output,
    LLMCallError
)
from app.rag_ingest import clean_documents, load_files, split_chunks
from app.rag_retrieval import (
    build_clarification_question,
    is_ambiguous_question,
    retrieve as retrieve_impl,
    rerank as rerank_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
    should_refuse,
    default_decompose_result,
    decompose_query_with_llm,
    retrieve_with_sub_queries
)
from app.rag_store import (
    build_current_manifest,
    build_index,
    get_index_cache_source as get_index_cache_source_impl,
    index_cache_exists as index_cache_exists_impl,
    load_index_cache as load_index_cache_impl,
    needs_rebuild as needs_rebuild_impl,
    save_index_cache as save_index_cache_impl,
    save_manifest,
)


# =========================
# 全局状态 / 常量
# =========================
logger = get_logger('chat_service')

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


def init_rag(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    """对外暴露的 RAG 初始化入口，内部复用重建索引流程。"""
    rebuild_index(data_dir, chunk_size, chunk_overlap)


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
        texts=texts,
        docs_registry=docs_registry,
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
# 查询阶段：总控
# =========================
def ask(
    question: str,
    top_k: int = 5,
    decompose_top_k: int = 10,
    retrieved_chunks: List[Dict[str, Any]] | None = None,
    trace_id : str = ""
):
    """执行完整 RAG 问答流程：澄清判断、可选拆分、重排、生成和引用绑定。"""
    question = (question or "").strip()
    # timings 只用于日志和性能观测，不参与业务判断。
    timings = {
        "decompose_cost_ms": 0,
        "rerank_cost_ms": 0,
        "retrieve_with_sub_queries_cost_ms": 0,
        "context_build_cost_ms": 0,
        "llm_cost_ms": 0,
        "parse_cost_ms": 0,
    }

    if is_ambiguous_question(question):
        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "final_chunks": [],
            "confidence": 0.0,
            "need_clarification": True,
            "clarification_question": build_clarification_question(question),
            "reason": "问题存在指代不明或缺少关键限定信息",
            "timings": timings
        }

    decompose_result = None
    decompose_cost_ms = 0

    if not Settings.ENABLE_QUERY_DECOMPOSE:
        decompose_result = default_decompose_result("query decomposition disabled")
        logger.info(
            f"[trace_id={trace_id}] [retrieve] decompose_skipped reason=disabled"
        )
    elif not Settings.DECOMPOSE_MODEL_NAME or not Settings.DECOMPOSE_API_URL:
        decompose_result = default_decompose_result("query decomposition not configured")
        logger.info(
            f"[trace_id={trace_id}] [retrieve] decompose_skipped reason=not_configured"
        )
    else:
        strat_decompose = time.perf_counter()
        try:
            decompose_result = decompose_query_with_llm(question)
        except Exception as e:
            logger.warning(
                f"[trace_id={trace_id}] [retrieve] decompose_failed fallback_to_single_query error={e}"
            )
            decompose_result = default_decompose_result("decompose failed, fallback to original question")

        decompose_cost_ms = int((time.perf_counter() - strat_decompose) * 1000)

    logger.info(
        f"[trace_id={trace_id}] [retrieve] decompose_cost_ms={decompose_cost_ms} "
        f"need_decompose={decompose_result.get('need_decompose')} "
        f"reason={decompose_result.get('reason', '')}"
    )
    timings["decompose_cost_ms"] = decompose_cost_ms

    if retrieved_chunks is None:
        retrieved_chunks = retrieve(question, top_k)

    start_rerank = time.perf_counter()

    if not decompose_result["need_decompose"]:
        candidates = retrieved_chunks
        reranked = rerank(candidates, question)
        final_chunks = reranked[:top_k]
        rerank_cost_ms = int((time.perf_counter() - start_rerank) * 1000)
        logger.info(f"[trace_id={trace_id}] [retrieve] rerank_cost_ms={rerank_cost_ms}")
        timings["rerank_cost_ms"] = rerank_cost_ms
    else:
        final_chunks = retrieve_with_sub_queries(
            question=question,
            sub_queries=decompose_result["sub_queries"],
            chunks=chunks,
            bm25=bm25,
            doc_embeddings=doc_embeddings,
            model=model,
            rerank_model=rerank_model,
            final_top_k=decompose_top_k
        )
        retrieve_with_sub_queries_cost_ms = int((time.perf_counter() - start_rerank) * 1000)
        logger.info(f"[trace_id={trace_id}] [retrieve] retrieve_with_sub_queries_cost_ms={retrieve_with_sub_queries_cost_ms}")
        timings["retrieve_with_sub_queries_cost_ms"] = retrieve_with_sub_queries_cost_ms

    hit_brief = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "rerank": round(chunk.get("rerank_score", 0), 4),
            "score": round(chunk.get("hybrid_score", 0), 4),
            "bm25": round(chunk.get("bm25_score", 0), 4),
            "vector": round(chunk.get("vector_score", 0), 4),
            "file_name": chunk.get("file_name"),
        }
        for chunk in final_chunks
    ]

    final_chunk_count = len(final_chunks)
    top_chunk_ids = [chunk.get("chunk_id") for chunk in final_chunks]
    top_scores = [
        round(chunk.get("hybrid_score", 0), 4)
        for chunk in final_chunks]

    logger.info(
        f"[trace_id={trace_id}] [retrieve] final_chunk_count={final_chunk_count} "
        f"top_chunk_ids={top_chunk_ids} top_scores={top_scores} hits={hit_brief}"
    )

    logger.info(f"[trace_id={trace_id}] [rag] 开始问答流程 top_k={top_k}")

    if should_refuse(final_chunks):
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "final_chunks": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "检索结果为空或 top1 分数低于阈值",
            "timings": timings
        }

    context_start = time.perf_counter()
    context = build_context(final_chunks, Settings.MAX_CONTEXT_CHARS)
    context_build_cost_ms = int((time.perf_counter() - context_start) * 1000)
    timings["context_build_cost_ms"] = context_build_cost_ms

    logger.info(
        f"[trace_id={trace_id}] [rag] context 构造完成 context_len={len(context)} "
        f"context_build_cost_ms={context_build_cost_ms}"
    )
    logger.info(f"[trace_id={trace_id}] [llm] 开始调用模型")

    llm_start = time.perf_counter()
    try:
        llm_out = call_llm(question, context, trace_id)
    except LLMCallError as e:
        timings["llm_cost_ms"] = int((time.perf_counter() - llm_start) * 1000)
        return {
            "status": "MODEL_ERROR",
            "answer": e.user_message,
            "references": [],
            "final_chunks": final_chunks,
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": e.user_message,
            "timings": timings
        }

    timings["llm_cost_ms"] = int((time.perf_counter() - llm_start) * 1000)

    logger.info(
        f"[trace_id={trace_id}] [llm] 模型调用完成 output_len={len(llm_out or '')}"
    )

    parse_start = time.perf_counter()
    parsed = parse_llm_output(llm_out)
    parse_cost_ms = int((time.perf_counter() - parse_start) * 1000)
    timings["parse_cost_ms"] = parse_cost_ms

    logger.info(
        f"[trace_id={trace_id}] [parse] 模型输出解析完成 ok={parsed.get('ok')} "
        f"parse_cost_ms={parse_cost_ms}"
    )

    if not parsed["ok"]:
        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "final_chunks": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型未按约定 JSON 格式返回",
            "timings": timings
        }

    final_references = format_reference(final_chunks, parsed["references"])

    if not final_references or STOP_WORD in parsed["answer"]:
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": parsed["answer"],
            "references": final_references,
            "final_chunks": final_chunks,
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型给出的答案没有足够的引用支撑",
            "timings": timings
        }

    confidence = estimate_confidence(
        retrieved_chunks=retrieved_chunks,
        final_references=final_references,
        status="OK"
    )

    return {
        "status": "OK",
        "answer": parsed["answer"],
        "references": final_references,
        "final_chunks": final_chunks,
        "confidence": confidence,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "检索命中有效证据，且答案成功绑定引用",
        "timings": timings
    }


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
