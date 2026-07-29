"""混合检索与重排模块。

融合 BM25、向量召回和交叉编码器重排，并支持多子查询结果合并。
"""

from typing import Any, Dict, List

import jieba
import numpy as np

from app.rag.chroma import ChromaUnavailableError, query_chroma
from app.common.logging import get_logger
from app.rag.policy import (
    build_clarification_question,
    is_ambiguous_question,
    should_refuse,
)
from app.rag.scoring import safe_float


__all__ = [
    "build_clarification_question",
    "clamp01",
    "is_ambiguous_question",
    "make_snippet",
    "minmax_score",
    "rerank",
    "retrieve",
    "retrieve_with_sub_queries",
    "safe_float",
    "search_relevant_chunks",
    "should_refuse",
    "tokenize",
]


logger = get_logger('chat_service')


# =========================
# 通用文本与分数工具
# =========================
def tokenize(text: str):
    """使用 jieba 对文本分词，供 BM25 检索使用。"""
    return list(jieba.cut(text))


def minmax_score(x: np.ndarray):
    """将一组分数归一化到 0 到 1 区间。"""
    x = x.astype(np.float32)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def clamp01(value: float) -> float:
    """将数值限制在 0 到 1 区间。"""
    value = float(value)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def make_snippet(text: str, max_len: int = 300) -> str:
    """去除文本空白并截取指定长度的预览片段。"""
    text = "".join((text or "").split())
    return text[:max_len]


# =========================
# 重排
# =========================
def rerank(rerank_model, candidates: list[dict], question: str=""):

    """使用交叉编码器对候选 chunk 按相关性重排。"""
    if not candidates:
        return []

    pairs = []
    for item in candidates:
        pairs.append((question, item["text"]))

    scores = rerank_model.predict(pairs)

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    candidates = sorted(
        candidates,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return candidates


# =========================
# 检索
# =========================
def _make_retrieve_result(
    chunk: dict,
    rank: int,
    bm25_score: float,
    vector_score: float,
    hybrid_score: float,
):
    """将 chunk 和分数组装成统一检索结果结构。"""
    metadata = chunk.get("metadata") or {}
    text = chunk.get("text", "") or ""

    return {
        "reference_id": f"Reference {rank}",
        "file_name": metadata.get("file_name", "unknown"),
        "source_path": metadata.get("source_path", "unknown"),
        "chunk_id": metadata.get("chunk_id", f"chunk_{rank}"),
        "position": metadata.get("position"),
        "text": text,
        "snippet": make_snippet(text),
        "bm25_score": float(bm25_score),
        "vector_score": float(vector_score),
        "hybrid_score": float(hybrid_score),
        "metadata": metadata,
    }


def _retrieve_with_local_embeddings(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
):
    """使用本地 numpy embedding 矩阵执行旧版混合召回。"""
    if doc_embeddings is None:
        return []

    tokenized_question = tokenize(question)
    bm25_score = bm25.get_scores(tokenized_question)

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    vector_score = doc_embeddings @ question_embedding

    bm25_norm = minmax_score(bm25_score)
    vector_norm = minmax_score(vector_score)
    hybrid_score = 0.4 * bm25_norm + 0.6 * vector_norm

    candidate_k = max(top_k * 5, top_k)
    top_indices = np.argsort(hybrid_score)[::-1][:candidate_k]

    raw_results = []
    for rank, idx in enumerate(top_indices, start=1):
        raw_results.append(
            _make_retrieve_result(
                chunk=chunks[idx],
                rank=rank,
                bm25_score=float(bm25_score[idx]),
                vector_score=float(vector_score[idx]),
                hybrid_score=float(hybrid_score[idx]),
            )
        )

    return raw_results


def _retrieve_with_chroma(
    question: str,
    chunks: list[dict],
    bm25,
    model,
    top_k: int = 5
):
    """使用 ChromaDB 做向量召回，并与 BM25 候选合并。"""
    candidate_k = max(top_k * 5, top_k)

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    chroma_hits = query_chroma(question_embedding, n_results=candidate_k)
    if not chroma_hits:
        return []

    tokenized_question = tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_question)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    chunk_by_id = {}
    for idx, chunk in enumerate(chunks):
        chunk_id = str((chunk.get("metadata") or {}).get("chunk_id", ""))
        if chunk_id:
            chunk_by_id[chunk_id] = (idx, chunk)

    candidates = {}

    for hit in chroma_hits:
        metadata = hit.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        idx, chunk = chunk_by_id.get(chunk_id, (None, None))

        if chunk is None:
            chunk = {
                "text": hit.get("text", "") or "",
                "metadata": metadata,
            }

        candidate_key = chunk_id or str(hit.get("record_id", ""))
        candidates[candidate_key] = {
            "chunk": chunk,
            "bm25_score": float(bm25_scores[idx]) if idx is not None else 0.0,
            "vector_score": float(hit.get("vector_score", 0.0)),
        }

    for idx in bm25_top_indices:
        idx = int(idx)
        chunk = chunks[idx]
        metadata = chunk.get("metadata") or {}
        chunk_id = str(metadata.get("chunk_id", idx))

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "chunk": chunk,
                "bm25_score": float(bm25_scores[idx]),
                "vector_score": 0.0,
            }

    if not candidates:
        return []

    candidate_items = list(candidates.values())
    bm25_values = np.array([item["bm25_score"] for item in candidate_items], dtype=np.float32)
    vector_values = np.array([item["vector_score"] for item in candidate_items], dtype=np.float32)

    bm25_norm = minmax_score(bm25_values)
    vector_norm = minmax_score(vector_values)

    for item, bm25_part, vector_part in zip(candidate_items, bm25_norm, vector_norm):
        item["hybrid_score"] = float(0.4 * bm25_part + 0.6 * vector_part)

    candidate_items.sort(key=lambda item: item["hybrid_score"], reverse=True)

    raw_results = []
    for rank, item in enumerate(candidate_items[:candidate_k], start=1):
        raw_results.append(
            _make_retrieve_result(
                chunk=item["chunk"],
                rank=rank,
                bm25_score=item["bm25_score"],
                vector_score=item["vector_score"],
                hybrid_score=item["hybrid_score"],
            )
        )

    return raw_results


def retrieve(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
):
    """综合 BM25 和向量相似度召回候选 chunk，优先使用 ChromaDB 向量库。"""
    if bm25 is None or model is None:
        return []

    question = (question or "").strip()
    if not question:
        return []

    try:
        chroma_results = _retrieve_with_chroma(
            question=question,
            chunks=chunks,
            bm25=bm25,
            model=model,
            top_k=top_k,
        )
        if chroma_results:
            return chroma_results
    except ChromaUnavailableError as e:
        logger.info(f"[retrieve] ChromaDB unavailable, fallback to local embeddings: {e}")
    except Exception as e:
        logger.warning(f"[retrieve] ChromaDB query failed, fallback to local embeddings: {e}")

    return _retrieve_with_local_embeddings(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=top_k,
    )

def retrieve_with_sub_queries(
    question,
    sub_queries,
    chunks,
    bm25,
    doc_embeddings,
    model,
    rerank_model,
    per_query_top_k=10,
    per_query_keep=2,
    final_top_k=10
):
    """多子问题召回时，保护每个子问题的高质量命中，再做全局补足。"""
    protected_chunks = {}
    all_candidates = {}

    # 原问题整体召回一次，保留全局语义。
    global_candidates = retrieve(
        question=question,
        chunks=chunks,
        bm25=bm25,
        doc_embeddings=doc_embeddings,
        model=model,
        top_k=5
    )

    for item in global_candidates:
        all_candidates[item["chunk_id"]] = item

    # 每个子查询单独召回、单独 rerank。
    for sub in sub_queries:
        sub_question = sub["sub_question"]
        search_query = sub["search_query"]

        candidates = retrieve(
            question=search_query,
            chunks=chunks,
            bm25=bm25,
            doc_embeddings=doc_embeddings,
            model=model,
            top_k=per_query_top_k
        )

        for item in candidates:
            all_candidates[item["chunk_id"]] = item

        reranked = rerank(
            candidates=candidates,
            question=sub_question,
            rerank_model=rerank_model
        )

        # 每个子问题至少保留 topN，避免某个子问题被全局排序完全挤掉。
        for item in reranked[:per_query_keep]:
            item["matched_sub_question"] = sub_question
            item["protected"] = True
            protected_chunks[item["chunk_id"]] = item

    merged_candidates = list(all_candidates.values())

    global_reranked = rerank(
        candidates=merged_candidates,
        question=question,
        rerank_model=rerank_model
    )

    final = []
    used_ids = set()

    for item in protected_chunks.values():
        final.append(item)
        used_ids.add(item["chunk_id"])

    for item in global_reranked:
        if item["chunk_id"] not in used_ids:
            item["protected"] = False
            final.append(item)
            used_ids.add(item["chunk_id"])

        if len(final) >= final_top_k:
            break

    for rank, item in enumerate(final, start=1):
        item["reference_id"] = f"Reference {rank}"

    return final


def search_relevant_chunks(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """对基础 retrieve 做薄封装，返回相关 chunk 列表。"""
    return retrieve(question, chunks, bm25, doc_embeddings, model, top_k)
