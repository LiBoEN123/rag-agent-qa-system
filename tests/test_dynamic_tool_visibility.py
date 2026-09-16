"""真实 create_agent.astream 的权限 schema、MCP 与隐藏调用回归。"""

from pydantic import RootModel

import asyncio
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.agent.framework_agent import build_framework_agent
from app.contracts.chat import ToolMode
from app.tools.catalog import get_local_tool_specs
from app.tools.context import ToolInvocationContext, build_default_tool_context
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools


class SchemaModel:
    def __init__(self, forced_call=None):
        self.schemas = []
        self.forced_call = forced_call

    def bind_tools(self, tools, **kwargs):
        self.schemas.append({tool.name: tool.tool_call_schema.model_json_schema() for tool in tools})
        return self

    async def ainvoke(self, messages):
        if self.forced_call and not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="", tool_calls=[self.forced_call])
        return AIMessage(content="done")


def stream(agent, context):
    inputs = {"messages": [("user", "check tool schemas")]}
    async def stream():
        output = None
        async for output in agent.astream(inputs, context=context, stream_mode="values"):
            pass
        return output

    return asyncio.run(stream())


def test_native_agent_binds_exact_schemas_for_each_context(monkeypatch, settings_factory):
    settings = settings_factory()
    tools = framework_tools.get_local_tools(settings)
    model = SchemaModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    agent = build_framework_agent(settings, tools=tools)
    kb = ["search_knowledge_base", "compare_documents", "list_documents", "get_chunk_text"]
    cases = [
        (build_default_tool_context(settings=settings), kb),
        (ToolInvocationContext(), []),
        (ToolInvocationContext(permissions=("knowledge_base:read",)), kb),
        (ToolInvocationContext(permissions=("web:search",)), ["web_search"]),
        (ToolInvocationContext(permissions=("web:fetch",)), ["web_fetch", "web_pdf_fetch"]),
        (ToolInvocationContext(permissions=("knowledge_base:read", "web:search")), ["search_knowledge_base", "web_search", *kb[1:]]),
        (ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read", "web:search", "web:fetch", "mcp:github")), ["search_knowledge_base"]),
        (ToolInvocationContext(tool_mode=ToolMode.FULL, permissions=("knowledge_base:read", "web:search")), ["search_knowledge_base", "web_search", *kb[1:]]),
    ]
    for context, expected in cases:
        model.schemas.clear()
        stream(agent, context)
        # 无可见工具时框架直接调用模型，不会调用 bind_tools。
        schemas = model.schemas[-1] if model.schemas else {}
        assert list(schemas) == expected
        assert schemas == {t.name: t.tool_call_schema.model_json_schema() for t in tools if t.name in expected}
        assert all("runtime" not in schema["properties"] for schema in schemas.values())
        assert all("tool_mode" not in schema["properties"] for schema in schemas.values())


def test_hidden_registered_tool_cannot_execute(monkeypatch, settings_factory):
    class Args(BaseModel):
        value: str

    calls = []
    hidden = ToolSpec(name="hidden_tool", version="1.0.0", description="private description", args_model=Args, output_model=RootModel[dict], required_permissions=("a:read",), handler=lambda args, context: calls.append(context))
    public = hidden.model_copy(update={"name": "public_tool", "required_permissions": ()})
    model = SchemaModel({"name": "hidden_tool", "args": {"value": "forged"}, "id": "hidden-call", "type": "tool_call"})
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    settings = settings_factory()
    output = stream(build_framework_agent(settings, tools=local_tools([hidden, public], settings)), ToolInvocationContext())
    message = next(m for m in output["messages"] if isinstance(m, ToolMessage))
    result = json.loads(message.content)
    assert all(list(s) == ["public_tool"] for s in model.schemas)
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["decision"] == "deny"
    assert result["runtime"]["attempt_count"] == 0
    assert calls == []


@pytest.mark.parametrize("server", ["github", "firecrawl"])
def test_mcp_base_tool_survives_local_filter(monkeypatch, settings_factory, server):
    external = StructuredTool.from_function(lambda value: value, name=f"{server}_read", description="MCP fixture")
    monkeypatch.setattr(framework_tools, "list_mcp_tools_for_context", lambda context, *, settings: [external])
    settings = settings_factory()
    context = ToolInvocationContext(permissions=(f"mcp:{server}",))
    tools = framework_tools.resolve_agent_tools(context, settings)
    model = SchemaModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    stream(build_framework_agent(settings, tools=tools), context)
    assert list(model.schemas[-1]) == [external.name]
    assert tools[-1] is external


def test_missing_runtime_fails_closed(settings_factory):
    calls = []

    class Args(BaseModel):
        value: str

    spec = ToolSpec(name="public_tool", version="1.0.0", description="public", args_model=Args, output_model=RootModel[dict], handler=lambda a, c: calls.append(c))
    tool = framework_tools.build_local_tool(spec, settings=settings_factory())
    with pytest.raises(ValueError, match="runtime"):
        tool.invoke({"value": "no runtime"})
    with pytest.raises(RuntimeError, match="ToolRuntime"):
        tool.func(runtime=None, value="missing")
    assert calls == []


def test_rag_only_filters_even_registered_mcp_tools(monkeypatch, settings_factory):
    settings = settings_factory()
    external = StructuredTool.from_function(lambda value: value, name="github_read", description="MCP fixture")
    model = SchemaModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    agent = build_framework_agent(settings, tools=[*framework_tools.get_local_tools(settings), external])
    stream(agent, ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read", "mcp:github")))
    assert list(model.schemas[-1]) == ["search_knowledge_base"]


@pytest.mark.parametrize(("name", "args"), [
    ("compare_documents", {"query": "test", "doc_a": "a.md", "doc_b": "b.md"}),
    ("web_search", {"query": "test"}),
])
def test_rag_only_forged_calls_are_denied_before_handler(monkeypatch, settings_factory, name, args):
    settings = settings_factory()
    handler = Mock(side_effect=AssertionError("forbidden handler executed"))
    specs = [spec.model_copy(update={"handler": handler}) if spec.name == name else spec
             for spec in get_local_tool_specs(settings)]
    model = SchemaModel({"name": name, "args": args, "id": "forged-call", "type": "tool_call"})
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    context = ToolInvocationContext(tool_mode=ToolMode.RAG_ONLY, permissions=("knowledge_base:read", "web:search"))
    output = stream(build_framework_agent(settings, tools=local_tools(specs, settings)), context)
    message = next(m for m in output["messages"] if isinstance(m, ToolMessage))
    result = json.loads(message.content)
    assert all(list(schema) == ["search_knowledge_base"] for schema in model.schemas)
    assert result["error_code"] == "policy_denied"
    assert result["handler_executed"] is False
    assert result["governance"]["decision"] == "deny"
    assert result["runtime"]["attempt_count"] == 0
    handler.assert_not_called()
