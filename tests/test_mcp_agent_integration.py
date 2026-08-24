"""create_agent 对 MCP Tool 的同步、异步与 Guardrail 集成测试。"""

import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent

from app.agent import framework_agent, framework_tools
from app.agent.streaming import project_agent_stream_part
from app.tools.audit import NullToolAuditSink
from app.tools.context import ToolInvocationContext
from app.tools.mcp.bridge import (
    McpToolCallInterceptor,
    build_governed_mcp_tool,
)
from app.tools.mcp.config import McpServerConfig
from app.tools.mcp.provider import (
    CachedMcpTool,
    McpToolProvider,
    install_mcp_tool_provider,
)
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from tests.support.agent_runtime import build_agent_harness


class FakeAgentModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tool_names = []
        self.bound_tools = []
        self.received_messages = []

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tools = list(tools)
        self.bound_tool_names = [tool.name for tool in self.bound_tools]
        return self

    def invoke(self, messages, **kwargs):
        del kwargs
        self.received_messages.append(list(messages))
        return self.responses.pop(0)

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)


def _server_config(**overrides):
    payload = {
        "transport": "http",
        "url": "https://weather.example.com/mcp",
        "allowed_tools": ["get_weather"],
    }
    payload.update(overrides)
    return McpServerConfig.model_validate(payload)


def _mcp_tool(settings, calls, server_config):
    interceptor = McpToolCallInterceptor(
        settings,
        server_configs={"weather": server_config},
        audit_sink=NullToolAuditSink(),
    )

    async def raw_coroutine(runtime=None, **args):
        request = MCPToolCallRequest(
            name="get_weather",
            args=args,
            server_name="weather",
            runtime=runtime,
        )

        async def handler(_request):
            calls.append((args["location"], runtime.context.trace_id))
            return CallToolResult(
                content=[TextContent(type="text", text="Sunny, 20 C")],
                structuredContent={"temperature": 20},
            )

        return await interceptor(request, handler), None

    raw = StructuredTool(
        name="weather_get_weather",
        description="Remote weather",
        args_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        coroutine=raw_coroutine,
        response_format="content_and_artifact",
    )
    return build_governed_mcp_tool(
        raw,
        server_alias="weather",
        remote_tool_name="get_weather",
        description="Safe weather",
        approval_required=("get_weather" in server_config.approval_required_tools),
    )


def _setup(monkeypatch, settings, model, calls, *, server_config=None):
    resolved_server_config = server_config or _server_config()
    tool = _mcp_tool(settings, calls, resolved_server_config)
    install_mcp_tool_provider(
        McpToolProvider(
            [
                CachedMcpTool(
                    server_alias="weather",
                    remote_tool_name="get_weather",
                    tool=tool,
                    server_config=resolved_server_config,
                )
            ],
            settings=settings,
        )
    )
    monkeypatch.setattr(
        framework_tools,
        "get_default_tool_provider",
        lambda: LocalToolProvider(ToolRegistry()),
    )
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    return build_agent_harness(settings), tool


def _input():
    return {
        "original_question": "北京天气如何",
        "effective_question": "北京天气如何",
        "top_k": 5,
        "decompose_top_k": 10,
        "decision": {"action": "agent"},
        "trace_id": "trace-mcp-agent",
        "session_id": "session-mcp-agent",
    }


def _context():
    return ToolInvocationContext(
        trace_id="trace-mcp-agent",
        session_id="session-mcp-agent",
        principal_id="mcp-agent-user",
        permissions=("mcp:weather",),
        source="mcp_agent_test",
    )


def _responses():
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather_get_weather",
                    "args": {"location": "Beijing"},
                    "id": "weather-call",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="北京晴，20 摄氏度。"),
    ]


def _assert_success(output, model, calls):
    assert model.bound_tool_names == ["weather_get_weather"]
    assert calls == [("Beijing", "trace-mcp-agent")]
    tool_message = next(
        message
        for messages in model.received_messages
        for message in messages
        if isinstance(message, ToolMessage)
    )
    payload = json.loads(tool_message.content)
    assert payload["ok"] is True
    assert payload["output"]["content_is_untrusted"] is True
    assert output["result"]["answer"] == "北京晴，20 摄氏度。"
    assert output["tool_call_count"] == 1
    assert output.get("knowledge_base_result") in (None, {})
    tool_trace = next(
        item
        for item in output["result"]["tool_trace"]
        if item.get("tool_name") == "weather_get_weather"
    )
    assert tool_trace["success"] is True
    assert tool_trace["attempt_count"] == 1


def test_mcp_tool_runs_through_sync_agent_and_native_async_bridge(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory()
    sync_calls = []
    sync_model = FakeAgentModel(_responses())
    sync_graph, source_tool = _setup(
        monkeypatch, settings, sync_model, sync_calls
    )
    sync_output = sync_graph.invoke(_input(), context=_context())
    _assert_success(sync_output, sync_model, sync_calls)
    assert sync_model.bound_tools == [source_tool]

    async_calls = []
    async_model = FakeAgentModel(_responses())
    _, async_tool = _setup(monkeypatch, settings, async_model, async_calls)
    runtime = type(
        "DirectRuntime",
        (),
        {"context": _context(), "tool_call_id": "async-weather-call"},
    )()
    async_result = asyncio.run(
        async_tool.coroutine(runtime=runtime, location="Beijing")
    )
    assert async_calls == [("Beijing", "trace-mcp-agent")]
    assert async_result["ok"] is True
    assert async_result["output"]["content_is_untrusted"] is True
    assert (
        async_result["output"]
        == json.loads(
            next(
                message
                for messages in sync_model.received_messages
                for message in messages
                if isinstance(message, ToolMessage)
            ).content
        )["output"]
    )


def test_default_service_principal_sees_and_executes_config_authorized_server(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory(
        tool_default_principal_id="chat_api",
        tool_default_permissions=("knowledge_base:read",),
    )
    calls = []
    model = FakeAgentModel(_responses())
    graph, _ = _setup(
        monkeypatch,
        settings,
        model,
        calls,
        server_config=_server_config(enabled_for_default_principal=True),
    )
    context = ToolInvocationContext(
        trace_id="trace-mcp-agent",
        session_id="session-mcp-agent",
        principal_id="chat_api",
        permissions=("knowledge_base:read",),
        source="mcp_agent_test",
    )

    output = graph.invoke(_input(), context=context)

    _assert_success(output, model, calls)


def test_mcp_external_write_stops_agent_for_human_approval(
    monkeypatch,
    settings_factory,
):
    settings = settings_factory()
    calls = []
    model = FakeAgentModel(_responses())
    graph, _ = _setup(
        monkeypatch,
        settings,
        model,
        calls,
        server_config=_server_config(approval_required_tools=["get_weather"]),
    )

    output = graph.invoke(_input(), context=_context())

    assert calls == []
    assert len(model.received_messages) == 1
    assert output["stop_reason"] == "approval_required"
    message = next(
        item
        for item in output["messages"]
        if isinstance(item, ToolMessage) and item.tool_call_id == "weather-call"
    )
    payload = json.loads(message.content)
    assert payload["error_code"] == "approval_required"
    assert payload["governance"]["risk_level"] == "external_write"
    assert payload["governance"]["approval_required"] is True
    assert payload["handler_executed"] is False


def test_mcp_native_async_result_projects_to_safe_stream_event(
    monkeypatch,
    settings_factory,
):
    """流式路径使用原生异步结果，SSE 投影不暴露参数或输出。"""
    settings = settings_factory()
    calls = []
    model = FakeAgentModel(_responses())
    _, tool = _setup(monkeypatch, settings, model, calls)
    runtime = type(
        "StreamRuntime",
        (),
        {"context": _context(), "tool_call_id": "stream-weather-call"},
    )()
    result = asyncio.run(tool.coroutine(runtime=runtime, location="Beijing"))
    assert calls == [("Beijing", "trace-mcp-agent")]
    assert result["ok"] is True

    projected = project_agent_stream_part(
        {
            "type": "updates",
            "ns": ("framework_agent:task",),
            "data": {
                "tools": {
                    "governance_events": [
                        {
                            "kind": "tool",
                            "name": "weather_get_weather",
                            "args": {"location": "Beijing"},
                            "result": result,
                            "ok": True,
                            "cost_ms": 7,
                            "tool_call_id": "stream-weather-call",
                        }
                    ]
                }
            },
        }
    )
    assert len(projected) == 1
    assert projected[0].event == "tool"
    assert projected[0].data == {
        "tool_name": "weather_get_weather",
        "phase": "completed",
        "status": "success",
        "cost_ms": 7,
    }
    serialized = json.dumps(projected[0].data, ensure_ascii=False)
    assert "Beijing" not in serialized
    assert "Sunny" not in serialized


def test_mcp_tool_runs_through_create_agent_astream(
    monkeypatch,
    settings_factory,
):
    """create_agent astream 通过原生 coroutine 完成同一 MCP Agent 循环。"""
    settings = settings_factory()
    calls = []
    model = FakeAgentModel(_responses())
    graph, _ = _setup(monkeypatch, settings, model, calls)

    async def collect():
        return [
            value
            async for value in graph.astream(
                _input(),
                context=_context(),
                stream_mode="values",
            )
        ]

    output = asyncio.run(collect())[-1]

    _assert_success(output, model, calls)


def test_repeated_mcp_calls_are_stopped_by_existing_guardrail(
    monkeypatch,
    settings_factory,
):
    calls = []
    repeated_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "weather_get_weather",
                    "args": {"location": "Beijing"},
                    "id": f"weather-call-{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index in range(1, 4)
    ]
    model = FakeAgentModel(repeated_responses)
    settings = settings_factory(agent_max_consecutive_same_tool_calls=2)
    graph, _ = _setup(monkeypatch, settings, model, calls)

    output = graph.invoke(_input(), context=_context())

    assert len(calls) == 2
    assert output["tool_call_count"] == 2
    assert output["stop_reason"] == "repeated_tool_call"
    blocked = next(
        message
        for message in output["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "weather-call-3"
    )
    assert json.loads(blocked.content)["handler_executed"] is False
