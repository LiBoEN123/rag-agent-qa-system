"""LangGraph RAG 子图构建与缓存测试。"""

import pytest

from app.rag.graph import builder as rag_graph_builder
from app.rag.graph import conditions as rag_graph_conditions
from app.rag.graph import nodes as rag_graph_nodes


@pytest.fixture(autouse=True)
def clear_rag_graph_cache():
    """隔离各测试使用的已编译 RAG Graph。"""
    rag_graph_builder.get_compiled_rag_graph.cache_clear()
    yield
    rag_graph_builder.get_compiled_rag_graph.cache_clear()


def test_build_and_compile_rag_graph_has_no_business_side_effects(monkeypatch):
    """创建和编译子图不会调用检索、模型、索引或工具。"""
    calls: list[str] = []

    def fail_if_called(*args, **kwargs):
        calls.append("business_call")
        raise AssertionError("RAG Graph compilation must be side-effect free")

    monkeypatch.setattr(rag_graph_nodes, "retrieve", fail_if_called)
    monkeypatch.setattr(rag_graph_nodes, "rerank", fail_if_called)
    monkeypatch.setattr(rag_graph_nodes, "decompose_query_with_llm", fail_if_called)
    monkeypatch.setattr(rag_graph_nodes, "call_llm", fail_if_called)

    uncompiled = rag_graph_builder.build_rag_graph()
    compiled = uncompiled.compile()

    assert calls == []
    assert {
        "validate_question",
        "decompose",
        "retrieve_and_rerank",
        "evidence_gate",
        "build_context",
        "generate",
        "parse_output",
        "bind_and_verify",
        "success_result",
    } <= set(compiled.get_graph().nodes)


def test_compiled_rag_graph_can_invoke_clarification_path(monkeypatch):
    """歧义问题从校验节点直接结束，不执行后续业务阶段。"""
    monkeypatch.setattr(
        rag_graph_nodes,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("澄清路径不应检索")
        ),
    )
    monkeypatch.setattr(
        rag_graph_nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("澄清路径不应调用模型")
        ),
    )

    output = rag_graph_builder.get_compiled_rag_graph().invoke(
        {
            "question": "这个？",
            "top_k": 5,
            "decompose_top_k": 10,
            "retrieved_chunks": None,
            "trace_id": "trace-clarify",
        }
    )

    assert output["result"]["status"] == "NEED_CLARIFICATION"
    assert output["result"]["need_clarification"] is True


def test_compiled_rag_graph_is_reused():
    """同一进程不会为每次 RAG 请求重复编译子图。"""
    first = rag_graph_builder.get_compiled_rag_graph()
    second = rag_graph_builder.get_compiled_rag_graph()

    assert first is second


def test_rag_graph_conditions_only_read_state():
    """条件函数仅根据 result 是否存在返回路由名称。"""
    unfinished = {"question": "测试问题"}
    finished = {"result": {"status": "OK"}}

    assert rag_graph_conditions.after_validate_condition(unfinished) == "decompose"
    assert rag_graph_conditions.after_validate_condition(finished) == "end"
    assert (
        rag_graph_conditions.after_evidence_gate_condition(unfinished)
        == "build_context"
    )
    assert rag_graph_conditions.after_generate_condition(unfinished) == "parse_output"
    assert rag_graph_conditions.after_parse_condition(unfinished) == "bind_and_verify"
    assert (
        rag_graph_conditions.after_citations_condition(unfinished)
        == "success_result"
    )
