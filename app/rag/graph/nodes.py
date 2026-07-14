"""LangGraph RAG 子图节点。

节点仅拆分旧 ``rag.ask()`` 的编排阶段，检索、重排、生成、引用和置信度
算法继续复用现有实现。
"""

import time
from typing import Any

from app.common.logging import get_logger
from app.config import Settings
from app.prompts import NO_ANSWER_FALLBACK
from app.rag.citations import verify_answer_citations
from app.rag.decomposition import (
    decompose_query_with_llm,
    default_decompose_result,
    should_decompose_query,
)
from app.rag.generation import (
    LLMCallError,
    build_context,
    call_llm,
    estimate_answer_confidence,
    estimate_retrieval_confidence,
    format_reference,
    parse_llm_output,
)
from app.rag.graph.state import RAGGraphState
from app.rag.pipeline import (
    STOP_WORD,
    rerank,
    retrieve,
    retrieve_with_sub_queries_from_current_index,
)
from app.rag.policy import (
    build_clarification_question,
    is_ambiguous_question,
    should_refuse,
)
from app.rag.results import (
    build_default_timings,
    build_evidence_gate_step,
    build_rag_result,
)


logger = get_logger("chat_service")


def _timings(state: RAGGraphState) -> dict[str, int | bool | str]:
    """读取当前 timings；Runtime 缺失输入时补齐默认字段。"""
    current = state.get("timings")
    if isinstance(current, dict):
        return dict(current)
    return build_default_timings()


def validate_question_node(state: RAGGraphState) -> dict[str, Any]:
    """清理问题并复用旧歧义判断生成澄清结果。"""
    question = (state.get("question") or "").strip()
    updates: dict[str, Any] = {"question": question}
    if not is_ambiguous_question(question):
        return updates

    timings = _timings(state)
    updates.update(
        {
            "terminal_reason": "ambiguous_question",
            "result": build_rag_result(
                status="NEED_CLARIFICATION",
                need_clarification=True,
                clarification_question=build_clarification_question(question),
                reason="问题存在指代不明或缺少关键限定信息",
                timings=timings,
            ),
        }
    )
    return updates


def decompose_node(state: RAGGraphState) -> dict[str, Any]:
    """按旧判断顺序执行可选 Query Decompose，并保留失败回退。"""
    question = state.get("question", "")
    trace_id = state.get("trace_id", "")
    timings = _timings(state)
    decompose_cost_ms = 0
    decompose_llm_used = False

    if not Settings.ENABLE_QUERY_DECOMPOSE:
        decompose_result = default_decompose_result("query decomposition disabled")
        timings["decompose_skipped_reason"] = "disabled"
        logger.info(
            "[trace_id=%s] [rag_graph] node=decompose skipped reason=disabled",
            trace_id,
        )
    else:
        should_decompose, decompose_rule_reason = should_decompose_query(question)
        timings["decompose_rule_reason"] = decompose_rule_reason

        if not should_decompose:
            decompose_result = default_decompose_result(
                f"decompose skipped by rule: {decompose_rule_reason}"
            )
            timings["decompose_skipped_reason"] = decompose_rule_reason
            logger.info(
                "[trace_id=%s] [rag_graph] node=decompose skipped reason=%s",
                trace_id,
                decompose_rule_reason,
            )
        elif not Settings.DECOMPOSE_MODEL_NAME or not Settings.DECOMPOSE_API_URL:
            decompose_result = default_decompose_result(
                "query decomposition not configured"
            )
            timings["decompose_skipped_reason"] = "not_configured"
            logger.info(
                "[trace_id=%s] [rag_graph] node=decompose skipped "
                "reason=not_configured",
                trace_id,
            )
        else:
            start_decompose = time.perf_counter()
            decompose_llm_used = True
            try:
                decompose_result = decompose_query_with_llm(question)
            except Exception as exc:
                logger.warning(
                    "[trace_id=%s] [rag_graph] node=decompose failed "
                    "fallback_to_single_query error=%s",
                    trace_id,
                    exc,
                )
                decompose_result = default_decompose_result(
                    "decompose failed, fallback to original question"
                )
            decompose_cost_ms = int(
                (time.perf_counter() - start_decompose) * 1000
            )

    timings["decompose_cost_ms"] = decompose_cost_ms
    timings["decompose_llm_used"] = decompose_llm_used
    logger.info(
        "[trace_id=%s] [rag_graph] node=decompose cost_ms=%s "
        "llm_used=%s need_decompose=%s reason=%s",
        trace_id,
        decompose_cost_ms,
        decompose_llm_used,
        decompose_result.get("need_decompose"),
        decompose_result.get("reason", ""),
    )
    return {
        "decompose_result": decompose_result,
        "decompose_llm_used": decompose_llm_used,
        "timings": timings,
    }


def retrieve_and_rerank_node(state: RAGGraphState) -> dict[str, Any]:
    """选择单查询重排或现有多查询检索门面，避免重复基础召回。"""
    question = state.get("question", "")
    trace_id = state.get("trace_id", "")
    top_k = int(state.get("top_k", Settings.TOP_K))
    decompose_top_k = int(
        state.get("decompose_top_k", Settings.DECOMPOSE_TOP_K)
    )
    decompose_result = state.get("decompose_result") or default_decompose_result()
    timings = _timings(state)
    start = time.perf_counter()

    if not decompose_result.get("need_decompose", False):
        candidates = state.get("retrieved_chunks")
        if candidates is None:
            candidates = retrieve(question, top_k)
        final_chunks = rerank(candidates, question)[:top_k]
        cost_ms = int((time.perf_counter() - start) * 1000)
        timings["rerank_cost_ms"] = cost_ms
        logger.info(
            "[trace_id=%s] [rag_graph] node=retrieve_and_rerank "
            "mode=single rerank_cost_ms=%s",
            trace_id,
            cost_ms,
        )
    else:
        final_chunks = retrieve_with_sub_queries_from_current_index(
            question=question,
            sub_queries=decompose_result.get("sub_queries", []),
            final_top_k=decompose_top_k,
        )
        cost_ms = int((time.perf_counter() - start) * 1000)
        timings["retrieve_with_sub_queries_cost_ms"] = cost_ms
        logger.info(
            "[trace_id=%s] [rag_graph] node=retrieve_and_rerank "
            "mode=multi retrieve_with_sub_queries_cost_ms=%s",
            trace_id,
            cost_ms,
        )

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
    logger.info(
        "[trace_id=%s] [rag_graph] node=retrieve_and_rerank "
        "final_chunk_count=%s top_chunk_ids=%s hits=%s",
        trace_id,
        len(final_chunks),
        [chunk.get("chunk_id") for chunk in final_chunks],
        hit_brief,
    )
    return {"final_chunks": final_chunks, "timings": timings}


def evidence_gate_node(state: RAGGraphState) -> dict[str, Any]:
    """复用现有置信度与拒答策略，在证据不足时直接终止。"""
    final_chunks = state.get("final_chunks") or []
    timings = _timings(state)
    retrieval_confidence = estimate_retrieval_confidence(final_chunks)
    updates: dict[str, Any] = {
        "retrieval_confidence": retrieval_confidence,
        "answer_llm_used": False,
    }
    if not should_refuse(final_chunks):
        return updates

    refuse_reason = "检索结果为空或证据分数低于阈值"
    evidence_gate = build_evidence_gate_step(
        final_chunks,
        refused=True,
        reason=refuse_reason,
    )
    updates.update(
        {
            "evidence_gate": evidence_gate,
            "terminal_reason": "insufficient_evidence",
            "result": build_rag_result(
                status="NO_RELEVANT_INFO",
                answer=NO_ANSWER_FALLBACK,
                retrieval_confidence=retrieval_confidence,
                reason=refuse_reason,
                timings=timings,
                decompose_llm_used=bool(
                    state.get("decompose_llm_used", False)
                ),
                evidence_gate=evidence_gate,
            ),
        }
    )
    return updates


def build_context_node(state: RAGGraphState) -> dict[str, Any]:
    """使用现有格式构造回答上下文，并仅记录长度和耗时。"""
    start = time.perf_counter()
    context = build_context(
        state.get("final_chunks") or [],
        Settings.MAX_CONTEXT_CHARS,
    )
    cost_ms = int((time.perf_counter() - start) * 1000)
    timings = _timings(state)
    timings["context_build_cost_ms"] = cost_ms
    logger.info(
        "[trace_id=%s] [rag_graph] node=build_context "
        "context_len=%s context_build_cost_ms=%s",
        state.get("trace_id", ""),
        len(context),
        cost_ms,
    )
    return {"context": context, "timings": timings}


def generate_node(state: RAGGraphState) -> dict[str, Any]:
    """通过统一回答模型入口生成 JSON 文本，并兼容现有模型错误。"""
    start = time.perf_counter()
    timings = _timings(state)
    updates: dict[str, Any] = {"answer_llm_used": True}
    try:
        llm_output = call_llm(
            state.get("question", ""),
            state.get("context", ""),
            state.get("trace_id", ""),
        )
    except LLMCallError as exc:
        timings["llm_cost_ms"] = int((time.perf_counter() - start) * 1000)
        updates.update(
            {
                "timings": timings,
                "terminal_reason": "model_error",
                "result": build_rag_result(
                    status="MODEL_ERROR",
                    answer=exc.user_message,
                    final_chunks=state.get("final_chunks") or [],
                    retrieval_confidence=float(
                        state.get("retrieval_confidence", 0.0)
                    ),
                    reason=exc.user_message,
                    timings=timings,
                    decompose_llm_used=bool(
                        state.get("decompose_llm_used", False)
                    ),
                    answer_llm_used=True,
                ),
            }
        )
        return updates

    timings["llm_cost_ms"] = int((time.perf_counter() - start) * 1000)
    logger.info(
        "[trace_id=%s] [rag_graph] node=generate output_len=%s",
        state.get("trace_id", ""),
        len(llm_output or ""),
    )
    updates.update({"llm_output": llm_output, "timings": timings})
    return updates


def parse_output_node(state: RAGGraphState) -> dict[str, Any]:
    """复用现有 JSON 解析协议，并将坏输出转换为兼容业务状态。"""
    start = time.perf_counter()
    parsed_output = parse_llm_output(state.get("llm_output", ""))
    cost_ms = int((time.perf_counter() - start) * 1000)
    timings = _timings(state)
    timings["parse_cost_ms"] = cost_ms
    updates: dict[str, Any] = {
        "parsed_output": parsed_output,
        "timings": timings,
    }
    logger.info(
        "[trace_id=%s] [rag_graph] node=parse_output ok=%s parse_cost_ms=%s",
        state.get("trace_id", ""),
        parsed_output.get("ok"),
        cost_ms,
    )
    if parsed_output.get("ok"):
        return updates

    updates.update(
        {
            "terminal_reason": "model_bad_output",
            "result": build_rag_result(
                status="MODEL_BAD_OUTPUT",
                answer=NO_ANSWER_FALLBACK,
                retrieval_confidence=float(
                    state.get("retrieval_confidence", 0.0)
                ),
                reason="模型未按约定 JSON 格式返回",
                timings=timings,
                decompose_llm_used=bool(
                    state.get("decompose_llm_used", False)
                ),
                answer_llm_used=bool(state.get("answer_llm_used", False)),
            ),
        }
    )
    return updates


def bind_and_verify_citations_node(state: RAGGraphState) -> dict[str, Any]:
    """把模型引用绑定到真实候选并复用现有引用支撑验证。"""
    final_chunks = state.get("final_chunks") or []
    parsed_output = state.get("parsed_output") or {}
    final_references = format_reference(
        final_chunks,
        parsed_output.get("references", []),
    )
    citation_verification = verify_answer_citations(
        question=state.get("question", ""),
        answer=parsed_output.get("answer", ""),
        references=final_references,
        min_overlap_ratio=0.1,
    )
    updates: dict[str, Any] = {
        "final_references": final_references,
        "citation_verification": citation_verification,
    }
    common = {
        "final_chunks": final_chunks,
        "retrieval_confidence": float(
            state.get("retrieval_confidence", 0.0)
        ),
        "timings": _timings(state),
        "decompose_llm_used": bool(state.get("decompose_llm_used", False)),
        "answer_llm_used": bool(state.get("answer_llm_used", False)),
        "citation_verification": citation_verification,
    }

    if not citation_verification.get("supported", False):
        updates.update(
            {
                "terminal_reason": "citation_verification_failed",
                "result": build_rag_result(
                    status="NO_RELEVANT_INFO",
                    answer=NO_ANSWER_FALLBACK,
                    references=final_references,
                    answer_confidence=0.0,
                    need_clarification=False,
                    clarification_question="",
                    reason=citation_verification.get("reason", ""),
                    **common,
                ),
            }
        )
        return updates

    answer = parsed_output.get("answer", "")
    if not final_references or STOP_WORD in answer:
        updates.update(
            {
                "terminal_reason": "insufficient_citation_support",
                "result": build_rag_result(
                    status="NO_RELEVANT_INFO",
                    answer=answer,
                    references=final_references,
                    reason="模型给出的答案没有足够的引用支撑",
                    **common,
                ),
            }
        )
    return updates


def success_result_node(state: RAGGraphState) -> dict[str, Any]:
    """复用现有答案置信度与结果构造器生成成功结果。"""
    final_chunks = state.get("final_chunks") or []
    final_references = state.get("final_references") or []
    retrieval_confidence = float(state.get("retrieval_confidence", 0.0))
    answer_confidence = estimate_answer_confidence(
        retrieved_chunks=final_chunks,
        final_references=final_references,
        status="OK",
        retrieval_confidence=retrieval_confidence,
    )
    parsed_output = state.get("parsed_output") or {}
    return {
        "terminal_reason": "success",
        "result": build_rag_result(
            status="OK",
            answer=parsed_output.get("answer", ""),
            references=final_references,
            final_chunks=final_chunks,
            retrieval_confidence=retrieval_confidence,
            answer_confidence=answer_confidence,
            reason="检索命中有效证据，且答案成功绑定引用",
            timings=_timings(state),
            decompose_llm_used=bool(state.get("decompose_llm_used", False)),
            answer_llm_used=bool(state.get("answer_llm_used", False)),
            citation_verification=state.get("citation_verification") or {},
        ),
    }


__all__ = [
    "bind_and_verify_citations_node",
    "build_context_node",
    "decompose_node",
    "evidence_gate_node",
    "generate_node",
    "parse_output_node",
    "retrieve_and_rerank_node",
    "success_result_node",
    "validate_question_node",
]
