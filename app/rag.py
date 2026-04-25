from typing import Any, Dict, List

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
)
from app.rag_ingest import clean_documents, load_files, split_chunks
from app.rag_retrieval import (
    build_clarification_question,
    is_ambiguous_question,
    retrieve as retrieve_impl,
    search_relevant_chunks as search_relevant_chunks_impl,
    should_refuse,
)
from app.rag_store import (
    build_current_manifest,
    build_index,
    index_cache_exists as index_cache_exists_impl,
    load_index_cache as load_index_cache_impl,
    needs_rebuild as needs_rebuild_impl,
    save_index_cache as save_index_cache_impl,
    save_manifest,
)


# =========================
# 全局状态 / 常量
# =========================
chunks: list[dict] = []
texts: list[str] = []
docs_registry: list[dict] = []

bm25 = None
model = None
doc_embeddings = None

RAG_READY = False
RAG_INIT_MESSAGE = "RAG 尚未初始化"
NO_DOCUMENTS_MESSAGE = "当前知识库为空，请先在 data 目录中放入文档后再提问"
STOPWORD = "我无法根据提供的文档确定答案"


# =========================
# 状态函数
# =========================
def get_rag_status():
    return {
        "ready": RAG_READY,
        "message": RAG_INIT_MESSAGE,
        "doc_count": len(docs_registry),
        "chunk_count": len(chunks)
    }


# =========================
# 初始化 / 构建
# =========================
def rebuild_index(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    global chunks, texts, docs_registry, bm25, model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE

    chunks = []
    texts = []
    docs_registry = []
    bm25 = None
    doc_embeddings = None
    model = None
    RAG_READY = False
    RAG_INIT_MESSAGE = "RAG 尚未初始化"

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

    index_bundle = build_index(all_chunks, model=model)

    chunks = all_chunks
    texts = index_bundle["texts"]
    bm25 = index_bundle["bm25"]
    doc_embeddings = index_bundle["doc_embeddings"]
    docs_registry = docs_registry_local
    model = index_bundle["model"]

    if not chunks:
        RAG_INIT_MESSAGE = "文档已加载，但切分后没有可用 chunk"
        return

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"


def init_rag(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    rebuild_index(data_dir, chunk_size, chunk_overlap)


def rebuild_and_persist_index(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
) -> tuple[bool, str]:
    rebuild_index(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not RAG_READY:
        return False, RAG_INIT_MESSAGE

    save_index_cache()
    save_manifest(build_current_manifest(data_dir, chunk_size, chunk_overlap))

    return True, "索引已重建，并已保存到磁盘"


# =========================
# cache / manifest
# =========================
def needs_rebuild(
    data_dir: str,
    chunk_size: int = Settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.CHUNK_OVERLAP
):
    return needs_rebuild_impl(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def index_cache_exists():
    return index_cache_exists_impl()


def save_index_cache():
    save_index_cache_impl(
        chunks=chunks,
        texts=texts,
        docs_registry=docs_registry,
        doc_embeddings=doc_embeddings
    )


def load_index_cache() -> tuple[bool, str]:
    global chunks, texts, docs_registry, bm25, model, doc_embeddings, RAG_READY, RAG_INIT_MESSAGE

    result = load_index_cache_impl(model=model)

    if not result["ok"]:
        RAG_READY = False
        RAG_INIT_MESSAGE = result["message"]
        return False, RAG_INIT_MESSAGE

    chunks = result["chunks"]
    texts = result["texts"]
    docs_registry = result["docs_registry"]
    doc_embeddings = result["doc_embeddings"]
    bm25 = result["bm25"]
    model = result["model"]

    if not chunks:
        RAG_READY = False
        RAG_INIT_MESSAGE = "索引缓存为空"
        return False, RAG_INIT_MESSAGE

    RAG_READY = True
    RAG_INIT_MESSAGE = "OK"
    return True, result["message"]


# =========================
# 查询阶段：检索包装
# =========================
def retrieve(question: str, top_k: int = 3):
    return retrieve_impl(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k
    )


def search_relevant_chunks(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
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
    top_k: int = 3,
    retrieved_chunks: List[Dict[str, Any]] | None = None
):
    if retrieved_chunks is None:
        retrieved_chunks = retrieve(question, top_k)

    if is_ambiguous_question(question):
        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "confidence": 0.0,
            "need_clarification": True,
            "clarification_question": build_clarification_question(question),
            "reason": "问题存在指代不明或缺少关键限定信息"
        }

    if should_refuse(retrieved_chunks):
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "检索结果为空或 top1 分数低于阈值"
        }

    context = build_context(retrieved_chunks, Settings.MAX_CONTEXT_CHARS)
    llm_out = call_llm(question, context)
    parsed = parse_llm_output(llm_out)

    if not parsed["ok"]:
        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": NO_ANSWER_FALLBACK,
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型未按约定 JSON 格式返回"
        }

    final_references = format_reference(retrieved_chunks, parsed["references"])

    if not final_references or STOPWORD in parsed["answer"]:
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": parsed["answer"],
            "references": final_references,
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "模型给出的答案没有足够的引用支撑"
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
        "confidence": confidence,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "检索命中有效证据，且答案成功绑定引用"
    }


# =========================
# 调试接口包装
# =========================
def preview_docs():
    return preview_docs_impl(docs_registry)


def preview_chunks(max_len:int, limit: int = 70, file_name: str | None = None):
    return preview_chunks_impl(chunks, max_len=max_len, limit=limit, file_name=file_name)


def get_chunk_full_text(chunk_id: int | str):
    return get_chunk_full_text_impl(chunks, chunk_id)