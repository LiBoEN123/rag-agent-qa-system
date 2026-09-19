"""证据策略模块。

供诊断搜索与表格诊断使用的检索分数门。
"""

from app.config import get_settings
from app.rag.scoring import safe_float


VECTOR_RELAX_RATIO = 0.70
RERANK_STRONG_THRESHOLD = 0.45
RERANK_VERY_STRONG_THRESHOLD = 0.55
HYBRID_USABLE_THRESHOLD = 0.20
HYBRID_STRONG_THRESHOLD = 0.35


def should_refuse(retrieved_chunks, vector_threshold: float = None):
    """根据向量、重排、混合和关键词分数判断是否应拒答。"""
    threshold = safe_float(
        vector_threshold
        if vector_threshold is not None
        else get_settings().vector_threshold
    )
    threshold = max(0.0, threshold)
    candidates = [
        item for item in (retrieved_chunks or []) if isinstance(item, dict)
    ]
    if not candidates:
        return True
    if max(safe_float(item.get("vector_score")) for item in candidates) >= threshold:
        return False

    relaxed_threshold = threshold * VECTOR_RELAX_RATIO
    for item in candidates[:3]:
        vector_score = safe_float(item.get("vector_score"))
        rerank_score = safe_float(item.get("rerank_score"))
        hybrid_score = safe_float(item.get("hybrid_score"))
        lexical_hit = safe_float(item.get("bm25_score")) > 0
        near_vector = vector_score >= relaxed_threshold
        strong_rerank = rerank_score >= RERANK_STRONG_THRESHOLD
        very_strong_rerank = rerank_score >= RERANK_VERY_STRONG_THRESHOLD
        usable_hybrid = hybrid_score >= HYBRID_USABLE_THRESHOLD
        strong_hybrid = hybrid_score >= HYBRID_STRONG_THRESHOLD
        if near_vector and (
            strong_rerank
            or strong_hybrid
            or (lexical_hit and usable_hybrid)
        ):
            return False
        if very_strong_rerank and strong_hybrid:
            return False
        if strong_rerank and lexical_hit and usable_hybrid:
            return False
        if item.get("protected") and (near_vector or strong_rerank):
            return False
    return True
