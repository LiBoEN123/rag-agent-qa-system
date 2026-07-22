"""ChatService 到真实 Agent Graph、Provider 和 ToolRuntime 的主路径测试。"""

import json
from importlib import import_module
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.agent import route_agent_action
from app.api.responses import CHAT_RESULT_RESPONSE_MAP
from app.graph import builder as graph_builder
from app.graph import nodes as graph_nodes
from app.graph.runtime import run_agent_langgraph
from app.observability.metrics import (
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
    normalize_metrics_status,
)
from app.services.chat_service import (
    ChatService,
    ChatServiceDependencies,
    ChatServiceRequest,
)
from app.tools import runtime as tool_runtime
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success


class ChatEchoArgs(BaseModel):
    text: str


class ChatAgentModel:
    def __init__(self):
        self.calls = 0
        self.received_messages: list[list] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.calls += 1
        self.received_messages.append(list(messages))
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "chat_echo",
                        "args": {"text": "完整主路径"},
                        "id": "chat-echo-call",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="ChatService 主路径完成。")


class ChatWebSearchModel(ChatAgentModel):
    def invoke(self, messages):
        self.calls += 1
        self.received_messages.append(list(messages))
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {
                            "query": "Python latest news",
                            "max_results": 2,
                            "freshness": "week",
                        },
                        "id": "chat-web-call",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(
            content="Python 新闻来源：https://example.com/python-news"
        )


class IncrementingClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        self.value += 0.001
        return self.value


def test_chat_service_runs_real_langgraph_provider_and_runtime(monkeypatch):
    graph_builder.reset_agent_graph_runtime()
    tool_runtime.reset_tool_runtime()
    registry = ToolRegistry()
    handler_contexts: list[ToolInvocationContext] = []
    handler_args: list[dict] = []
    runtime_calls: list[tuple[str, ToolInvocationContext]] = []

    def handler(args, context):
        handler_args.append(dict(args))
        handler_contexts.append(context)
        return tool_success("chat_echo", args, {"text": args["text"]})

    registry.register(
        ToolSpec(
            name="chat_echo",
            version="1.0.0",
            description="回显 ChatService 集成测试文本",
            args_schema={"text": "string"},
            output_schema={"text": "string"},
            handler=handler,
            args_model=ChatEchoArgs,
        )
    )
    model = ChatAgentModel()
    original_execute = tool_runtime.ToolRuntime.execute

    def recording_execute(self, tool_name, tool_args=None, *, context=None):
        runtime_calls.append((tool_name, context))
        return original_execute(self, tool_name, tool_args, context=context)

    monkeypatch.setattr(tool_runtime.ToolRuntime, "execute", recording_execute)
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", 6)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_CALLS", 10)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_ERRORS", 3)
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS",
        2,
    )
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS",
        2,
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOTAL_COST_MS", 120000)
    monkeypatch.setattr(graph_nodes.Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_DEFAULT_PRINCIPAL_ID", "chat-user")
    monkeypatch.setattr(
        graph_nodes.Settings,
        "TOOL_DEFAULT_PERMISSIONS",
        ("knowledge_base:read",),
    )
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_AUDIT_ENABLED", False)

    session_writes: list[tuple[str, str, str]] = []
    metric_writes: list[dict] = []
    bad_cases: list[dict] = []
    clock = IncrementingClock()

    def rewrite(question, history, trace_id=""):
        return {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "need_check": False,
            "need_rewrite": False,
            "reason": "integration passthrough",
            "rewrite_llm_used": False,
            "error": "",
        }

    dependencies = ChatServiceDependencies(
        settings=SimpleNamespace(TOP_K=5, DECOMPOSE_TOP_K=10),
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
        route_agent_action=route_agent_action,
        get_rag_status=lambda: {"ready": True, "message": "OK"},
        run_agent_langgraph=run_agent_langgraph,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        normalize_metrics_status=normalize_metrics_status,
        log_and_write_metrics=lambda record: metric_writes.append(dict(record)),
        write_bad_cases=lambda record: bad_cases.append(dict(record)),
        append_message=lambda session_id, role, content: session_writes.append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-07-20T10:00:00.000",
        perf_counter=clock,
        response_map=CHAT_RESULT_RESPONSE_MAP,
    )
    service = ChatService(dependencies)

    try:
        outcome = service.handle(
            ChatServiceRequest(
                question="请走完整 Agent 工具主路径",
                top_k=5,
                decompose_top_k=10,
                session_id="session-chat-integration",
                trace_id="trace-chat-integration",
                request_enter_time="2026-07-20T09:59:59.000",
                request_enter_perf=99.0,
            )
        )
    finally:
        graph_builder.reset_agent_graph_runtime()
        tool_runtime.reset_tool_runtime()

    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.result["status"] == "OK"
    assert outcome.result["answer"] == "ChatService 主路径完成。"
    assert model.calls == 2
    assert model.bound_tool_names == ["chat_echo"]
    assert isinstance(model.received_messages[1][-1], ToolMessage)
    tool_payload = json.loads(model.received_messages[1][-1].content)
    assert tool_payload["output"] == {"text": "完整主路径"}
    assert runtime_calls[0][0] == "chat_echo"
    assert handler_args == [{"text": "完整主路径"}]
    assert handler_contexts == [runtime_calls[0][1]]
    context = handler_contexts[0]
    assert context.trace_id == "trace-chat-integration"
    assert context.session_id == "session-chat-integration"
    assert context.principal_id == "chat-user"
    assert context.permissions == ("knowledge_base:read",)
    assert context.source == "chat_api"
    assert session_writes == [
        ("session-chat-integration", "user", "请走完整 Agent 工具主路径"),
        (
            "session-chat-integration",
            "assistant",
            "ChatService 主路径完成。",
        ),
    ]
    assert len(metric_writes) == 1
    assert bad_cases == []


def test_chat_service_completes_web_search_through_production_chain(monkeypatch):
    web_search_module = import_module("app.tools.web_search")
    graph_builder.reset_agent_graph_runtime()
    tool_runtime.reset_tool_runtime()
    ddgs_calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            ddgs_calls.append(("init", kwargs))

        def text(self, query, **kwargs):
            ddgs_calls.append(("text", query, kwargs))
            return [
                {
                    "title": "Python release news",
                    "href": "https://example.com/python-news",
                    "body": "A new Python release is available.",
                }
            ]

    model = ChatWebSearchModel()
    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", 6)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_CALLS", 10)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_ERRORS", 3)
    monkeypatch.setattr(
        graph_nodes.Settings, "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS", 2
    )
    monkeypatch.setattr(
        graph_nodes.Settings, "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS", 2
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOTAL_COST_MS", 120000)
    monkeypatch.setattr(graph_nodes.Settings, "LANGGRAPH_CHECKPOINTER", "disabled")
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_DEFAULT_PRINCIPAL_ID", "web-user")
    monkeypatch.setattr(
        graph_nodes.Settings,
        "TOOL_DEFAULT_PERMISSIONS",
        ("knowledge_base:read", "web:search"),
    )
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_AUDIT_ENABLED", False)

    session_writes = []
    metric_writes = []
    rewrite_calls = []
    clock = IncrementingClock()

    def rewrite(question, history, trace_id=""):
        rewrite_calls.append((question, list(history), trace_id))
        return {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "need_check": False,
            "need_rewrite": False,
            "reason": "integration passthrough",
            "rewrite_llm_used": False,
            "error": "",
        }

    dependencies = ChatServiceDependencies(
        settings=SimpleNamespace(TOP_K=5, DECOMPOSE_TOP_K=10),
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
        route_agent_action=route_agent_action,
        get_rag_status=lambda: {"ready": True, "message": "OK"},
        run_agent_langgraph=run_agent_langgraph,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        normalize_metrics_status=normalize_metrics_status,
        log_and_write_metrics=lambda record: metric_writes.append(dict(record)),
        write_bad_cases=lambda record: None,
        append_message=lambda session_id, role, content: session_writes.append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-07-22T10:00:00.000",
        perf_counter=clock,
        response_map=CHAT_RESULT_RESPONSE_MAP,
    )

    try:
        outcome = ChatService(dependencies).handle(
            ChatServiceRequest(
                question="今天有哪些重要的 Python 新闻？",
                top_k=5,
                decompose_top_k=10,
                session_id="session-chat-web",
                trace_id="trace-chat-web",
                request_enter_time="2026-07-22T09:59:59.000",
                request_enter_perf=99.0,
            )
        )
    finally:
        graph_builder.reset_agent_graph_runtime()
        tool_runtime.reset_tool_runtime()

    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.result["status"] == "OK"
    assert outcome.result["references"] == []
    assert outcome.result["action"] == "agent"
    assert model.calls == 2
    assert "web_search" in model.bound_tool_names
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "chat-web-call"
    assert json.loads(message.content)["output"]["results"][0]["url"] == (
        "https://example.com/python-news"
    )
    assert len(ddgs_calls) == 2
    assert rewrite_calls == [
        ("今天有哪些重要的 Python 新闻？", [], "trace-chat-web")
    ]
    assert session_writes == [
        ("session-chat-web", "user", "今天有哪些重要的 Python 新闻？"),
        (
            "session-chat-web",
            "assistant",
            "Python 新闻来源：https://example.com/python-news",
        ),
    ]
    assert len(metric_writes) == 1
    assert metric_writes[0]["tool_called"] is True
    assert metric_writes[0]["tool_success"] is True
    assert metric_writes[0]["tool_name"] == "web_search"
