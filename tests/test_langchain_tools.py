"""LangChain Tool 适配层测试。"""

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.tools import (
    get_langchain_tool,
    get_langchain_tool_map,
    get_langchain_tool_names,
    get_langchain_tools,
)
from app.tools.registry import tool_registry
from app.tools.context import ToolInvocationContext
from app.tools.schemas import (
    CompareDocumentsArgs,
    GetChunkTextArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    SimpleCalculatorArgs,
)


CORE_TOOL_MODELS = {
    "search_knowledge_base": SearchKnowledgeBaseArgs,
    "compare_documents": CompareDocumentsArgs,
    "simple_calculator": SimpleCalculatorArgs,
    "list_documents": ListDocumentsArgs,
    "get_chunk_text": GetChunkTextArgs,
}


def test_langchain_tool_collection_matches_registry():
    """工具列表、映射、顺序和描述均以注册表为事实来源。"""
    tools = get_langchain_tools()
    registry_names = tool_registry.names()
    names = [tool.name for tool in tools]

    assert isinstance(tools, list)
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert names == registry_names
    assert len(names) == len(set(names))
    assert set(CORE_TOOL_MODELS) <= set(names)
    assert get_langchain_tool_names() == registry_names
    assert list(get_langchain_tool_map()) == registry_names

    for tool in tools:
        spec = tool_registry.get(tool.name)
        assert spec is not None
        assert tool.description
        assert tool.description == spec.description


@pytest.mark.parametrize("tool_name,args_model", CORE_TOOL_MODELS.items())
def test_langchain_tools_reuse_registered_pydantic_models(tool_name, args_model):
    """LangChain args_schema 直接复用 ToolSpec 中原有的 Pydantic 模型。"""
    tool = get_langchain_tool(tool_name)
    spec = tool_registry.get(tool_name)

    assert tool is not None
    assert spec is not None
    assert tool.args_schema is args_model
    assert tool.args_schema is spec.args_model
    assert "trace_id" not in tool.args


@pytest.mark.parametrize(
    "tool_name,invalid_args",
    [
        ("search_knowledge_base", {}),
        ("compare_documents", {"query": "主题", "doc_a": "文档甲"}),
        ("simple_calculator", {}),
        ("get_chunk_text", {}),
    ],
)
def test_missing_required_args_do_not_dispatch(monkeypatch, tool_name, invalid_args):
    """缺少必填参数时由 LangChain/Pydantic 拒绝，且不进入项目分发器。"""
    calls = []
    monkeypatch.setattr(
        "app.tools.langchain_tools.dispatch_tool",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    tool = get_langchain_tool(tool_name)
    assert tool is not None
    with pytest.raises(ValidationError):
        tool.invoke(invalid_args)

    assert calls == []


@pytest.mark.parametrize(
    "tool_name,tool_args",
    [
        ("search_knowledge_base", {"query": "测试问题", "top_k": 0}),
        (
            "compare_documents",
            {"query": "主题", "doc_a": "文档甲", "doc_b": "文档乙", "top_k": 11},
        ),
    ],
)
def test_invalid_top_k_does_not_dispatch(monkeypatch, tool_name, tool_args):
    """非法 top_k 保持现有模型约束，并在分发前失败。"""
    calls = []
    monkeypatch.setattr(
        "app.tools.langchain_tools.dispatch_tool",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    tool = get_langchain_tool(tool_name)
    assert tool is not None
    with pytest.raises(ValidationError):
        tool.invoke(tool_args)

    assert calls == []


def test_schema_defaults_empty_args_and_chunk_id_types(monkeypatch):
    """默认参数、空参数工具及 chunk_id 联合类型保持现有 Schema 行为。"""
    calls = []

    def fake_dispatch(tool_name, tool_args=None, trace_id=""):
        calls.append((tool_name, tool_args, trace_id))
        return {"ok": True}

    monkeypatch.setattr("app.tools.langchain_tools.dispatch_tool", fake_dispatch)

    get_langchain_tool_map()["search_knowledge_base"].invoke({"query": "测试"})
    get_langchain_tool_map()["compare_documents"].invoke(
        {"query": "主题", "doc_a": "文档甲", "doc_b": "文档乙"}
    )
    get_langchain_tool_map()["list_documents"].invoke({})
    get_langchain_tool_map()["get_chunk_text"].invoke({"chunk_id": 7})
    get_langchain_tool_map()["get_chunk_text"].invoke({"chunk_id": "chunk-7"})

    assert calls == [
        ("search_knowledge_base", {"query": "测试", "top_k": 5}, ""),
        (
            "compare_documents",
            {"query": "主题", "doc_a": "文档甲", "doc_b": "文档乙", "top_k": 3},
            "",
        ),
        ("list_documents", {}, ""),
        ("get_chunk_text", {"chunk_id": 7}, ""),
        ("get_chunk_text", {"chunk_id": "chunk-7"}, ""),
    ]


@pytest.mark.parametrize("ok", [True, False])
def test_langchain_tool_forwards_once_and_returns_dispatch_result(monkeypatch, ok):
    """适配器仅调用一次分发器，并原样返回成功或失败字典。"""
    calls = []
    expected = {
        "status": "success" if ok else "error",
        "ok": ok,
        "tool_name": "simple_calculator",
        "tool_input": {"expression": "1 + 2"},
        "output": {"result": 3} if ok else None,
        "error": "" if ok else "mock failure",
    }

    def fake_dispatch(tool_name, tool_args=None, trace_id=""):
        calls.append((tool_name, tool_args, trace_id))
        return expected

    monkeypatch.setattr("app.tools.langchain_tools.dispatch_tool", fake_dispatch)

    tool = get_langchain_tool("simple_calculator", trace_id="test-trace")
    assert tool is not None
    result = tool.invoke({"expression": "1 + 2"})

    assert result is expected
    assert calls == [
        ("simple_calculator", {"expression": "1 + 2"}, "test-trace")
    ]
    assert "trace_id" not in calls[0][1]


def test_trace_id_is_isolated_between_tool_sets(monkeypatch):
    """不同请求构建的 Tool 闭包不会串用 trace_id。"""
    calls = []

    def fake_dispatch(tool_name, tool_args=None, trace_id=""):
        calls.append((tool_name, tool_args, trace_id))
        return {"status": "success", "ok": True}

    monkeypatch.setattr("app.tools.langchain_tools.dispatch_tool", fake_dispatch)

    tools_a = get_langchain_tool_map(trace_id="trace-a")
    tools_b = get_langchain_tool_map(trace_id="trace-b")
    tools_a["simple_calculator"].invoke({"expression": "1 + 1"})
    tools_b["simple_calculator"].invoke({"expression": "2 + 2"})
    tools_a["simple_calculator"].invoke({"expression": "3 + 3"})

    assert [call[2] for call in calls] == ["trace-a", "trace-b", "trace-a"]


def test_unknown_langchain_tool_returns_none_and_registry_behavior_is_unchanged():
    """适配查询返回 None，不改变注册表原有未知工具错误协议。"""
    assert get_langchain_tool("unknown_tool") is None

    result = tool_registry.dispatch("unknown_tool", {"value": 1})
    assert result["status"] == "error"
    assert result["ok"] is False
    assert result["tool_name"] == "unknown_tool"
    assert result["tool_input"] == {"value": 1}
    assert result["output"] is None
    assert result["error"] == "unknown tool: unknown_tool"
    assert result["error_code"] == "unknown_tool"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 0


def test_langchain_tool_forwards_complete_governance_context(monkeypatch):
    calls = []
    context = ToolInvocationContext(
        trace_id="context-trace",
        request_id="request-1",
        principal_id="service-1",
        permissions=("knowledge_base:read",),
        source="langchain-test",
    )

    def fake_dispatch(tool_name, tool_args=None, trace_id="", *, context=None):
        calls.append((tool_name, tool_args, trace_id, context))
        return {"status": "success", "ok": True}

    monkeypatch.setattr("app.tools.langchain_tools.dispatch_tool", fake_dispatch)
    tool = get_langchain_tool("list_documents", context=context)
    assert tool is not None
    tool.invoke({})
    assert calls == [("list_documents", {}, "", context)]
