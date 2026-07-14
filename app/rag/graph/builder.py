"""LangGraph RAG 子图构建器。"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.rag.graph.conditions import (
    after_citations_condition,
    after_evidence_gate_condition,
    after_generate_condition,
    after_parse_condition,
    after_validate_condition,
)
from app.rag.graph.nodes import (
    bind_and_verify_citations_node,
    build_context_node,
    decompose_node,
    evidence_gate_node,
    generate_node,
    parse_output_node,
    retrieve_and_rerank_node,
    success_result_node,
    validate_question_node,
)
from app.rag.graph.state import RAGGraphState


def build_rag_graph(
) -> StateGraph[RAGGraphState, None, RAGGraphState, RAGGraphState]:
    """创建尚未编译的 RAG StateGraph，不触发检索、模型或索引加载。"""
    builder = StateGraph(RAGGraphState)
    builder.add_node("validate_question", validate_question_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    builder.add_node("evidence_gate", evidence_gate_node)
    builder.add_node("build_context", build_context_node)
    builder.add_node("generate", generate_node)
    builder.add_node("parse_output", parse_output_node)
    builder.add_node("bind_and_verify", bind_and_verify_citations_node)
    builder.add_node("success_result", success_result_node)

    builder.add_edge(START, "validate_question")
    builder.add_conditional_edges(
        "validate_question",
        after_validate_condition,
        {"decompose": "decompose", "end": END},
    )
    builder.add_edge("decompose", "retrieve_and_rerank")
    builder.add_edge("retrieve_and_rerank", "evidence_gate")
    builder.add_conditional_edges(
        "evidence_gate",
        after_evidence_gate_condition,
        {"build_context": "build_context", "end": END},
    )
    builder.add_edge("build_context", "generate")
    builder.add_conditional_edges(
        "generate",
        after_generate_condition,
        {"parse_output": "parse_output", "end": END},
    )
    builder.add_conditional_edges(
        "parse_output",
        after_parse_condition,
        {"bind_and_verify": "bind_and_verify", "end": END},
    )
    builder.add_conditional_edges(
        "bind_and_verify",
        after_citations_condition,
        {"success_result": "success_result", "end": END},
    )
    builder.add_edge("success_result", END)
    return builder


@lru_cache(maxsize=1)
def get_compiled_rag_graph(
) -> CompiledStateGraph[RAGGraphState, None, RAGGraphState, RAGGraphState]:
    """返回进程内可复用的已编译 RAG 子图。"""
    return build_rag_graph().compile()


__all__ = ["build_rag_graph", "get_compiled_rag_graph"]
