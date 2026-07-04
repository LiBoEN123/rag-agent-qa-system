from app.tool_registry import tool_registry
from app.tools import dispatch_tool


def test_tool_registry_names_include_builtin_tools():
    """验证内置工具已注册到统一注册中心。"""
    names = tool_registry.names()

    assert "simple_calculator" in names
    assert "list_documents" in names
    assert "get_chunk_text" in names


def test_tool_registry_router_specs_shape():
    """验证 Router 可读取注册工具描述。"""
    specs = tool_registry.router_specs()

    assert isinstance(specs, list)
    assert specs

    for spec in specs:
        assert "name" in spec
        assert "description" in spec
        assert "args_schema" in spec


def test_tool_registry_dispatch_simple_calculator():
    """验证注册中心可以调用计算器工具。"""
    result = tool_registry.dispatch(
        "simple_calculator",
        {"expression": "1 + 2 * 3"}
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 7


def test_tool_registry_dispatch_unknown_tool():
    """验证未知工具返回兼容错误结构。"""
    result = tool_registry.dispatch("unknown_tool", {})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["tool_name"] == "unknown_tool"
    assert result["tool_input"] == {}
    assert result["output"] is None
    assert result["error"] == "unknown tool: unknown_tool"


def test_dispatch_tool_keeps_compatible_entrypoint(monkeypatch, tmp_path):
    """验证旧的 dispatch_tool 入口仍可调用注册工具。"""
    import app.tools as tools

    monkeypatch.setattr(
        tools,
        "TOOL_CALLS_PATH",
        tmp_path / "tool_calls.jsonl",
        raising=False
    )

    result = dispatch_tool(
        "simple_calculator",
        {"expression": "2 + 3"},
        trace_id="test_trace"
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["output"]["result"] == 5
