"""真实 create_agent.invoke/astream 的权限 schema、MCP 与隐藏调用回归。"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.agent.framework_agent import build_framework_agent
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

    def invoke(self, messages):
        if self.forced_call and not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="", tool_calls=[self.forced_call])
        return AIMessage(content="done")

    async def ainvoke(self, messages):
        return self.invoke(messages)


def invoke(agent, context, mode):
    inputs = {"messages": [("user", "check tool schemas")]}
    if mode == "sync":
        return agent.invoke(inputs, context=context)

    async def stream():
        output = None
        async for output in agent.astream(inputs, context=context, stream_mode="values"):
            pass
        return output

    return asyncio.run(stream())


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("sql_enabled", [False, True])
def test_native_agent_binds_exact_schemas_for_each_context(monkeypatch, settings_factory, mode, sql_enabled, sql_business_db):
    settings = settings_factory(sql_tool_enabled=sql_enabled, sql_database_url=f"sqlite:///{sql_business_db}", sql_allowed_tables=("students",))
    tools = framework_tools.get_local_tools(settings)
    model = SchemaModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    agent = build_framework_agent(settings, tools=tools)
    kb = ["search_knowledge_base", "compare_documents", "simple_calculator", "list_documents", "get_chunk_text"]
    cases = [
        (build_default_tool_context(settings=settings), kb),
        (ToolInvocationContext(), ["simple_calculator"]),
        (ToolInvocationContext(permissions=("knowledge_base:read",)), kb),
        (ToolInvocationContext(permissions=("web:search",)), ["web_search", "simple_calculator"]),
        (ToolInvocationContext(permissions=("web:fetch",)), ["web_fetch", "web_pdf_fetch", "simple_calculator"]),
        (ToolInvocationContext(permissions=("database:read",)), ["simple_calculator"] + (["list_sql_tables", "describe_sql_table", "query_sql_readonly"] if sql_enabled else [])),
        (ToolInvocationContext(permissions=("knowledge_base:read", "web:search")), ["search_knowledge_base", "web_search", *kb[1:]]),
    ]
    for context, expected in cases:
        invoke(agent, context, mode)
        schemas = model.schemas[-1]
        assert list(schemas) == expected
        assert schemas == {t.name: t.tool_call_schema.model_json_schema() for t in tools if t.name in expected}
        assert all("runtime" not in schema["properties"] for schema in schemas.values())


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_hidden_registered_tool_cannot_execute(monkeypatch, settings_factory, mode):
    class Args(BaseModel):
        value: str

    calls = []
    hidden = ToolSpec(name="hidden_tool", version="1.0.0", description="private description", args_model=Args, output_schema={}, required_permissions=("a:read",), handler=lambda args, context: calls.append(context))
    public = hidden.model_copy(update={"name": "public_tool", "required_permissions": ()})
    model = SchemaModel({"name": "hidden_tool", "args": {"value": "forged"}, "id": "hidden-call", "type": "tool_call"})
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    settings = settings_factory()
    output = invoke(build_framework_agent(settings, tools=local_tools([hidden, public], settings)), ToolInvocationContext(), mode)
    message = next(m for m in output["messages"] if isinstance(m, ToolMessage))
    result = json.loads(message.content)
    assert all(list(s) == ["public_tool"] for s in model.schemas)
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["decision"] == "deny"
    assert result["runtime"]["attempt_count"] == 0
    assert calls == []


@pytest.mark.parametrize("server", ["github", "firecrawl"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_mcp_base_tool_survives_local_filter(monkeypatch, settings_factory, server, mode):
    external = StructuredTool.from_function(lambda value: value, name=f"{server}_read", description="MCP fixture")
    monkeypatch.setattr(framework_tools, "list_mcp_tools_for_context", lambda context: [external])
    settings = settings_factory()
    context = ToolInvocationContext(permissions=(f"mcp:{server}",))
    tools = framework_tools.resolve_agent_tools(context, settings)
    model = SchemaModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    invoke(build_framework_agent(settings, tools=tools), context, mode)
    assert list(model.schemas[-1]) == ["simple_calculator", external.name]
    assert tools[-1] is external


def test_missing_runtime_fails_closed(settings_factory):
    calls = []

    class Args(BaseModel):
        value: str

    spec = ToolSpec(name="public_tool", version="1.0.0", description="public", args_model=Args, output_schema={}, handler=lambda a, c: calls.append(c))
    tool = framework_tools.build_local_tool(spec, settings=settings_factory())
    with pytest.raises(ValueError, match="runtime"):
        tool.invoke({"value": "no runtime"})
    with pytest.raises(RuntimeError, match="ToolRuntime"):
        tool.func(runtime=None, value="missing")
    assert calls == []
