"""混合检索与重排模块。

对内存中的全库 BM25、Embedding 分数融合排序，再交由交叉编码器重排。
"""

from typing import Any, Dict, List

import jieba
import numpy as np

__all__ = [
    "clamp01",
    "make_snippet",
    "minmax_score",
    "rerank",
    "retrieve",
    "search_relevant_chunks",
    "tokenize",
]


# =========================
# 通用文本与分数工具
# =========================
def tokenize(text: str):
    """使用 jieba 对文本分词，供 BM25 检索使用。"""
    return list(jieba.cut(text))


def minmax_score(x: np.ndarray):
    """将一组分数归一化到 0 到 1 区间。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("分数必须是一维有限数值数组")
    if not x.size:
        return x
    minimum, maximum = float(x.min()), float(x.max())
    if np.allclose(minimum, maximum):
        return np.zeros_like(x)
    # float64 中间值避免极端有限分数相减时溢出；只分配 O(N) 空间。
    return ((x.astype(np.float64) - minimum) / (maximum - minimum)).astype(np.float32)


def clamp01(value: float) -> float:
    """将数值限制在 0 到 1 区间。"""
    value = float(value)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def make_snippet(text: str, max_len: int = 300) -> str:
    """将连续空白规范化为空格，并截取指定长度的预览片段。"""
    text = " ".join((text or "").split())
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


def _cosine_scores(doc_embeddings: np.ndarray, question_embedding: np.ndarray):
    """批量计算余弦；兼容历史非单位向量，不复制或归一化整张文档矩阵。"""
    if question_embedding.ndim != 1 or question_embedding.size != doc_embeddings.shape[1]:
        raise ValueError("查询 embedding 必须是一维向量，且维度与文档 embeddings 匹配")

    # 使用 float64 累加平方，避免有效 float32 向量的范数溢出或下溢。
    # einsum 不产生 N×D 的平方临时矩阵；范数也能检测 NaN/Inf 和零向量。
    doc_norms = np.sqrt(np.einsum("ij,ij->i", doc_embeddings, doc_embeddings, dtype=np.float64))
    query_norm = np.sqrt(np.einsum("i,i->", question_embedding, question_embedding, dtype=np.float64))
    if not np.isfinite(doc_norms).all() or np.any(doc_norms == 0):
        raise ValueError("文档 embeddings 包含非有限数值或零向量")
    if not np.isfinite(query_norm) or query_norm == 0:
        raise ValueError("查询 embedding 包含非有限数值或零向量")

    # 当前模型已做 L2 Normalize；除以范数同时兼容旧向量和注入的模型。
    query_unit = np.asarray(question_embedding / query_norm, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (doc_embeddings @ query_unit) / doc_norms
    if not np.isfinite(scores).all():
        raise ValueError("余弦相似度计算产生非有限分数")
    return np.clip(scores, -1.0, 1.0)


def retrieve(
    question: str,
    chunks: list[dict],
    bm25,
    doc_embeddings,
    model,
    top_k: int = 5
):
    """按全库 BM25 和余弦分数统一 Min-Max 融合，返回供重排使用的候选。"""
    question = (question or "").strip()
    if not question or not chunks or top_k <= 0:
        return []

    if doc_embeddings is None:
        raise ValueError("检索缺少必需的文档 embeddings 矩阵")
    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    if doc_embeddings.ndim != 2 or doc_embeddings.shape[1] == 0:
        raise ValueError("文档 embeddings 必须是非空二维矩阵")
    if doc_embeddings.shape[0] != len(chunks):
        raise ValueError("文档 embedding 数量与 chunk 数量不一致")
    if bm25 is None or model is None:
        raise ValueError("检索缺少必需的 BM25 索引或 embedding 模型")

    # runtime 按 chunks 顺序建立 BM25；加载器将 chunks 和文档向量成对排序。
    bm25_scores = np.asarray(bm25.get_scores(tokenize(question)))
    if bm25_scores.shape != (len(chunks),):
        raise ValueError("BM25 分数必须是一维数组，且数量与 chunk 数量一致")
    bm25_norm = minmax_score(bm25_scores)

    question_embedding = np.asarray(model.embed_query(question), dtype=np.float32)
    vector_scores = _cosine_scores(doc_embeddings, question_embedding)
    vector_norm = minmax_score(vector_scores)
    hybrid_scores = 0.4 * bm25_norm + 0.6 * vector_norm

    candidate_k = min(max(top_k * 5, top_k), len(chunks))
    # 对负分数稳定升序排序，使并列项保持原始 chunks 顺序。
    top_indices = np.argsort(-hybrid_scores, kind="stable")[:candidate_k]

    raw_results = []
    for rank, idx in enumerate(top_indices, start=1):
        raw_results.append(
            _make_retrieve_result(
                chunk=chunks[idx],
                rank=rank,
                bm25_score=float(bm25_scores[idx]),
                vector_score=float(vector_scores[idx]),
                hybrid_score=float(hybrid_scores[idx]),
            )
        )

    return raw_results


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
