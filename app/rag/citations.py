"""答案引用校验模块。

提取答案与证据关键词，检查引用是否足以支撑最终生成内容。
"""

import re
import unicodedata
from typing import Any


REFERENCE_STOPWORDS = {
    "这个", "那个", "什么", "怎么", "如何", "为什么", "是否", "可以", "需要",
    "用户", "问题", "回答", "根据", "文档", "内容", "相关", "说明",
    "的是", "一个", "以及", "进行", "通过", "如果", "时候", "没有",
    "请问", "哪些", "多少", "一种", "一些",
    "the", "and", "for", "with", "from", "this", "that", "what", "how",
    "are", "was", "were", "you", "your", "can", "could", "should", "would",
    "about", "into", "over", "under", "between", "than", "then", "there",
}

REFUSAL_ANSWER_PATTERNS = (
    "我无法根据提供的文档确定答案",
    "没有检索到足够相关的知识库内容，无法根据当前文档回答。",
)


def extract_reference_keywords(text: str) -> set[str]:
    """从问题、答案或引用文本中提取用于粗略匹配的关键词集合。"""
    text = unicodedata.normalize("NFKC", text or "").lower()

    keywords = set()

    # 英文、数字、函数名，例如 createapp、mount、v-model、computed。
    english_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-]{1,}|\d+(?:\.\d+)?", text)
    for token in english_tokens:
        token = (token or "").strip().lower()
        if token and token not in REFERENCE_STOPWORDS:
            keywords.add(token)

    # 中文连续片段用二字滑窗做粗粒度匹配，避免引入额外分词依赖。
    chinese_blocks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for block in chinese_blocks:
        for i in range(len(block) - 1):
            word = block[i: i + 2]
            if word and word not in REFERENCE_STOPWORDS:
                keywords.add(word)

    return keywords


def is_refusal_answer(answer: str) -> bool:
    """判断答案是否为系统拒答，拒答不需要引用支撑校验。"""
    answer = (answer or "").strip()
    return any(pattern in answer for pattern in REFUSAL_ANSWER_PATTERNS)


def collect_reference_text(references: list[dict]) -> str:
    """从 references 中提取可验证的证据文本。"""
    parts = []
    for ref in references or []:
        if isinstance(ref, dict):
            value: Any = (
                ref.get("snippet")
                or ref.get("text")
                or ref.get("content")
                or ""
            )
        else:
            value = ref

        text = str(value or "").strip()
        if text:
            parts.append(text)

    return "\n".join(parts)


def build_verification_result(
    supported: bool,
    reason: str,
    overlap_ratio: float = 0.0,
    matched_keywords: set[str] | None = None,
    expected_keyword_count: int = 0,
    evidence_keyword_count: int = 0,
) -> dict:
    """构造稳定的引用验证返回结构。"""
    return {
        "supported": bool(supported),
        "reason": reason,
        "overlap_ratio": round(float(overlap_ratio), 4),
        "matched_keywords": sorted(matched_keywords or set()),
        "expected_keyword_count": int(expected_keyword_count),
        "evidence_keyword_count": int(evidence_keyword_count),
    }


def verify_answer_citations(
    question: str,
    answer: str,
    references: list[dict],
    min_overlap_ratio: float = 0.2,
) -> dict:
    """验证答案是否被绑定后的引用片段支撑。"""
    question = (question or "").strip()
    answer = (answer or "").strip()

    if not question:
        return build_verification_result(
            supported=False,
            reason="question is empty",
        )

    if not answer:
        return build_verification_result(
            supported=False,
            reason="answer is empty",
        )

    if is_refusal_answer(answer):
        return build_verification_result(
            supported=True,
            reason="refusal answer does not require citation verification",
        )

    if not references:
        return build_verification_result(
            supported=False,
            reason="references 为空，无法验证答案依据",
        )

    evidence_text = collect_reference_text(references)
    if not evidence_text.strip():
        return build_verification_result(
            supported=False,
            reason="references 存在，但 snippet/text/content 为空，无法验证答案依据",
        )

    expected_keywords = extract_reference_keywords(f"{question}\n{answer}")
    evidence_keywords = extract_reference_keywords(evidence_text)
    overlap_keywords = expected_keywords & evidence_keywords
    overlap_ratio = len(overlap_keywords) / max(len(expected_keywords), 1)

    if not expected_keywords:
        return build_verification_result(
            supported=False,
            reason="未提取到可用于验证的答案关键词",
            overlap_ratio=overlap_ratio,
            matched_keywords=overlap_keywords,
            expected_keyword_count=len(expected_keywords),
            evidence_keyword_count=len(evidence_keywords),
        )

    if overlap_ratio >= min_overlap_ratio:
        return build_verification_result(
            supported=True,
            reason="",
            overlap_ratio=overlap_ratio,
            matched_keywords=overlap_keywords,
            expected_keyword_count=len(expected_keywords),
            evidence_keyword_count=len(evidence_keywords),
        )

    return build_verification_result(
        supported=False,
        reason="答案与引用片段关键词重合度过低",
        overlap_ratio=overlap_ratio,
        matched_keywords=overlap_keywords,
        expected_keyword_count=len(expected_keywords),
        evidence_keyword_count=len(evidence_keywords),
    )
