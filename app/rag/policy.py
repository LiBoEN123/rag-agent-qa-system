"""证据策略模块。

根据多路检索分数判断是否拒答，并为模糊问题生成针对性的澄清追问。
"""

from app.config import Settings
from app.rag.scoring import safe_float


AMBIGUOUS_WORDS = [
    "这个", "那个", "它", "这条", "这项", "这份",
    "上面", "前面", "后面", "这里", "那里",
]
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
        else Settings.VECTOR_THRESHOLD
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


def is_ambiguous_question(question: str) -> bool:
    """判断问题是否过短或包含缺少上下文的指代表达。"""
    normalized = (question or "").strip()
    return (
        not normalized
        or len(normalized) <= 3
        or (
            any(word in normalized for word in AMBIGUOUS_WORDS)
            and len(normalized) < 18
        )
    )


def build_clarification_question(question: str) -> str:
    """根据问题主题生成更具体的中文澄清追问。"""
    normalized = (question or "").strip()
    if "退款" in normalized:
        return "你想问的是退款条件、退款时间窗口，还是退款申请方式？"
    if "报销" in normalized:
        return "你指的是哪一类费用？比如交通费、住宿费、餐费，还是办公用品？"
    if "流程" in normalized:
        return "你想查询哪一类流程？比如退款流程、报销流程、请假流程，还是采购流程？"
    return "你的问题还不够具体，请补充对象、场景或范围后再问。"
