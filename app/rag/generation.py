"""RAG 答案生成模块。

负责上下文组装、回答模型调用、JSON 解析、引用绑定和置信度估算。
"""

import json
from typing import List

from app.config import get_settings
from app.llm.factory import get_answer_model
from app.llm.gateway import ModelGateway
from app.prompts import SYSTEM_PROMPT_B, USER_PROMPT, NO_ANSWER_FALLBACK
from app.rag.retrieval import clamp01, make_snippet, safe_float


# =========================
# 上下文构造
# =========================
def build_context(retrieved_chunks, max_chars: int | None=None):
    """把检索结果整理成 LLM 提示词中的 evidence context。"""
    if max_chars is None:
        max_chars = get_settings().max_context_chars

    parts = []
    total = 0

    for item in retrieved_chunks:
        snippet = item["text"]
        block = (
            f"[{item['reference_id']}]\n"
            f"file_name:{item['file_name']}\n"
            f"score:{item['hybrid_score']}\n"
            f"chunk_id:{item['chunk_id']}\n"
            f"snippet:{snippet}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


def _build_answer_messages(question: str, context: str) -> list[dict[str, str]]:
    """使用原有 Prompt 内容构造统一消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_B},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                context=context,
                question=question,
            ),
        },
    ]


def call_llm(question: str, context: str, trace_id: str = "") -> str:
    """通过统一 Gateway 调用回答模型。"""
    gateway = ModelGateway(
        role="answer",
        langchain_model=get_answer_model(),
    )
    return gateway.invoke_text(
        _build_answer_messages(question, context),
        trace_id=trace_id,
    )


# =========================
# 模型输出处理
# =========================
def parse_llm_output(llm_output: str):
    """解析模型返回的 JSON，提取 answer 和 references。"""
    llm_output = llm_output.strip()

    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(llm_output)
        answer = data.get("answer", NO_ANSWER_FALLBACK)
        references = data.get("references", [])
        if not isinstance(references, list):
            references = []

        return {
            "ok": True,
            "answer": answer,
            "references": references
        }
    except Exception:
        return {
            "ok": False,
            "answer": NO_ANSWER_FALLBACK,
            "references": []
        }


def format_reference(retrieved_chunks, selected_reference: list):
    """将模型选择的引用编号映射回真实 chunk 元数据。"""
    ref_map = {item["reference_id"]: item for item in retrieved_chunks}

    final_refs = []
    for ref_id in selected_reference:
        if ref_id in ref_map:
            item = ref_map[ref_id]
            final_refs.append({
                "reference_id": item["reference_id"],
                "file_name": item["file_name"],
                "source_path": item["source_path"],
                "chunk_id": item["chunk_id"],
                "position": item.get("position"),
                "snippet": make_snippet(item["text"]),
                "score": round(float(item.get("hybrid_score", 0.0)), 4),
                "rerank_score":round(float(item.get("rerank_score", 0.0)), 4)
            })

    return final_refs

def chunk_evidence_score(item: dict) -> float:
    """把单个 chunk 的多路检索信号压成 0 到 1 的证据强度。"""
    vector_score = clamp01(safe_float(item.get("vector_score")))
    rerank_score = clamp01(safe_float(item.get("rerank_score")))
    hybrid_score = clamp01(safe_float(item.get("hybrid_score", item.get("score", 0.0))))
    lexical_hit = 1.0 if safe_float(item.get("bm25_score")) > 0 else 0.0

    weighted_score = (
        0.50 * vector_score
        + 0.30 * rerank_score
        + 0.15 * hybrid_score
        + 0.05 * lexical_hit
    )

    return round(
        clamp01(
            max(
                weighted_score,
                0.85 * vector_score,
                0.75 * rerank_score if (hybrid_score >= 0.20 or lexical_hit) else 0.0,
            )
        ),
        4
    )


def estimate_retrieval_confidence(retrieved_chunks: List[dict]) -> float:
    """根据最终候选证据质量估算检索置信度。"""
    candidates = [
        item
        for item in retrieved_chunks
        if isinstance(item, dict)
    ]
    if not candidates:
        return 0.0

    evidence_scores = sorted(
        (chunk_evidence_score(item) for item in candidates),
        reverse=True
    )
    top_score = evidence_scores[0]
    second_score = evidence_scores[1] if len(evidence_scores) > 1 else 0.0
    score_gap = clamp01(top_score - second_score) if len(evidence_scores) > 1 else 0.0

    supporting_count = sum(
        1
        for score in evidence_scores[:5]
        if score >= 0.35
    )
    support_ratio = clamp01(supporting_count / 3)

    retrieval_score = 0.70 * top_score + 0.20 * support_ratio + 0.10 * score_gap
    return round(clamp01(retrieval_score), 3)


def estimate_answer_confidence(
    retrieved_chunks: List[dict],
    final_references: List[dict],
    status: str,
    retrieval_confidence: float | None = None
) -> float:
    """结合检索质量和引用支撑度估算最终回答置信度。"""
    if status != "OK" or not retrieved_chunks or not final_references:
        return 0.0

    if retrieval_confidence is None:
        retrieval_confidence = estimate_retrieval_confidence(retrieved_chunks)
    retrieval_confidence = clamp01(retrieval_confidence)

    reference_ids = {
        str(ref.get("reference_id", ""))
        for ref in final_references
        if isinstance(ref, dict)
    }
    cited_chunks = [
        item
        for item in retrieved_chunks
        if str(item.get("reference_id", "")) in reference_ids
    ]

    cited_scores = [
        chunk_evidence_score(item)
        for item in cited_chunks
    ]
    citation_quality = (
        sum(cited_scores) / len(cited_scores)
        if cited_scores
        else 0.0
    )
    reference_ratio = clamp01(
        len(final_references) / max(1, min(len(retrieved_chunks), 5))
    )

    answer_score = (
        0.60 * retrieval_confidence
        + 0.25 * citation_quality
        + 0.15 * reference_ratio
    )
    answer_score = min(answer_score, retrieval_confidence + 0.10)
    return round(clamp01(answer_score), 3)
