"""LangGraph RAG 子图的无副作用条件边。"""

from typing import Literal

from app.rag.graph.state import RAGGraphState


def _has_result(state: RAGGraphState) -> bool:
    """判断当前状态是否已经生成合法终止结果。"""
    return isinstance(state.get("result"), dict)


def after_validate_condition(
    state: RAGGraphState,
) -> Literal["decompose", "end"]:
    """歧义问题直接结束，其余问题进入拆分判断。"""
    return "end" if _has_result(state) else "decompose"


def after_evidence_gate_condition(
    state: RAGGraphState,
) -> Literal["build_context", "end"]:
    """证据不足时结束，证据可用时构造上下文。"""
    return "end" if _has_result(state) else "build_context"


def after_generate_condition(
    state: RAGGraphState,
) -> Literal["parse_output", "end"]:
    """模型调用失败时结束，否则规范化结构化输出。"""
    return "end" if _has_result(state) else "parse_output"


def after_parse_condition(
    state: RAGGraphState,
) -> Literal["bind_and_verify", "end"]:
    """结构化输出不合法时结束，否则绑定并验证引用。"""
    return "end" if _has_result(state) else "bind_and_verify"


def after_citations_condition(
    state: RAGGraphState,
) -> Literal["success_result", "end"]:
    """引用不充分时结束，否则构造成功结果。"""
    return "end" if _has_result(state) else "success_result"


__all__ = [
    "after_citations_condition",
    "after_evidence_gate_condition",
    "after_generate_condition",
    "after_parse_condition",
    "after_validate_condition",
]
