"""ToolProvider 发现与可见性行为测试。"""

from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry


class TextArgs(BaseModel):
    """Provider 测试参数。"""

    text: str


def _build_spec(name: str, calls: list[str]) -> ToolSpec:
    """构造会记录实际 handler 执行的只读测试工具。"""

    def handler(args, context):
        calls.append(args["text"])
        return {
            "status": "success",
            "ok": True,
            "tool_name": name,
            "tool_input": args,
            "output": {"text": args["text"]},
            "error": "",
        }

    return ToolSpec(
        name=name,
        version="1.0.0",
        description=f"返回 {name} 收到的测试文本",
        output_schema={"text": "string"},
        handler=handler,
        args_model=TextArgs,
    )


def test_local_provider_returns_registered_specs_without_execution():
    """Provider 保持 Registry 顺序和对象身份，发现阶段不执行工具。"""
    registry = ToolRegistry()
    calls: list[str] = []
    first = _build_spec("first_text", calls)
    second = _build_spec("second_text", calls)
    registry.register(first)
    registry.register(second)
    provider = LocalToolProvider(registry)

    specs = provider.list_specs(context=ToolInvocationContext())

    assert specs == [first, second]
    assert [spec.name for spec in specs] == ["first_text", "second_text"]
    assert specs[0].description == first.description
    assert specs[0].args_model is TextArgs
    assert calls == []


def test_default_local_provider_exposes_only_permitted_production_tools(
    tool_registry,
):
    """默认 Provider 以生产 Registry 为来源并隐藏缺权工具。"""
    provider = LocalToolProvider(tool_registry)

    names = [
        spec.name
        for spec in provider.list_specs(
            context=ToolInvocationContext(permissions=("knowledge_base:read",))
        )
    ]

    assert set(names) < set(tool_registry.names())
    assert "web_search" in tool_registry.names()
    assert "web_fetch" in tool_registry.names()
    assert "web_pdf_fetch" in tool_registry.names()
    assert "web_search" not in names
    assert "web_fetch" not in names
    assert "web_pdf_fetch" not in names
    assert set(names) == {
        "search_knowledge_base",
        "compare_documents",
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
    }


def test_local_provider_does_not_cache_request_context():
    """同一 Provider 可安全服务不同请求，Context 不进入实例状态。"""
    registry = ToolRegistry()
    registry.register(_build_spec("isolated_text", []))
    provider = LocalToolProvider(registry)
    context_a = ToolInvocationContext(trace_id="trace-a", source="request-a")
    context_b = ToolInvocationContext(trace_id="trace-b", source="request-b")

    assert provider.list_specs(context=context_a) == provider.list_specs(
        context=context_b
    )
    assert not hasattr(provider, "context")
