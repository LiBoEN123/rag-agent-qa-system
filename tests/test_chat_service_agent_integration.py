"""ChatService 到真实 Agent Graph、Provider 和 ToolRuntime 的主路径测试。"""

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
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
    def __init__(self):
        super().__init__()
        self.selected_url = ""

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
        latest = messages[-1]
        assert isinstance(latest, ToolMessage)
        payload = json.loads(latest.content)
        if self.calls == 2:
            self.selected_url = payload["output"]["results"][0]["url"]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_fetch",
                        "args": {"url": self.selected_url, "max_chars": 4000},
                        "id": "chat-fetch-call",
                        "type": "tool_call",
                    }
                ],
            )
        output = payload["output"]
        return AIMessage(
            content=(
                f"{output['title']}：{output['content']} "
                f"来源：{output['final_url']}"
            )
        )


class ChatWebPdfModel(ChatAgentModel):
    """在 ChatService 主路径中搜索 PDF、读取页范围并回答。"""

    def __init__(self):
        super().__init__()
        self.selected_url = ""

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
                            "query": "governed PDF fixture",
                            "max_results": 2,
                            "freshness": "any",
                        },
                        "id": "chat-pdf-search-call",
                        "type": "tool_call",
                    }
                ],
            )
        latest = messages[-1]
        assert isinstance(latest, ToolMessage)
        payload = json.loads(latest.content)
        if self.calls == 2:
            self.selected_url = payload["output"]["results"][0]["url"]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_pdf_fetch",
                        "args": {
                            "url": self.selected_url,
                            "start_page": 1,
                            "max_pages": 2,
                            "max_chars": 4000,
                        },
                        "id": "chat-pdf-fetch-call",
                        "type": "tool_call",
                    }
                ],
            )
        output = payload["output"]
        return AIMessage(
            content=(
                f"{output['title']}，相关页码 {output['page_numbers']}，"
                f"来源：{output['final_url']}"
            )
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
    routed_questions: list[str] = []

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
    original_route = graph_nodes.route_agent_action

    def recording_route(question):
        routed_questions.append(question)
        return original_route(question)

    monkeypatch.setattr(graph_nodes, "route_agent_action", recording_route)
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
            "rewritten_question": "请执行改写后的完整 Agent 工具主路径",
            "history_used": True,
            "rewritten": True,
            "need_check": True,
            "need_rewrite": True,
            "reason": "integration rewrite",
            "rewrite_llm_used": False,
            "error": "",
        }

    dependencies = ChatServiceDependencies(
        settings=SimpleNamespace(TOP_K=5, DECOMPOSE_TOP_K=10),
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
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
    assert outcome.question == "请走完整 Agent 工具主路径"
    assert routed_questions == ["请执行改写后的完整 Agent 工具主路径"]
    assert model.calls == 2
    assert model.received_messages[0][0].content == (
        "请执行改写后的完整 Agent 工具主路径"
    )
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
    decide_trace = next(
        step
        for step in outcome.result["tool_trace"]
        if step.get("step") == "decide"
    )
    assert metric_writes[0]["route_cost_ms"] == decide_trace["route_cost_ms"]
    assert bad_cases == []


def test_chat_service_completes_web_search_and_fetch_through_production_chain(
    monkeypatch,
):
    web_search_module = import_module("app.tools.web_search")
    web_download_module = import_module("app.tools.web_download")
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

    article_html = """
    <html><head><title>Python release news</title></head>
    <body><main><h1>Python release news</h1>
    <p>A new Python release is available with runtime improvements.</p>
    </main></body></html>
    """

    class FakeResponse:
        status_code = 200
        encoding = "utf-8"
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __init__(self, url):
            self.url = url
            self.closed = False

        def iter_content(self, chunk_size):
            del chunk_size
            yield article_html.encode()

        def close(self):
            self.closed = True

    class FakeSession:
        def __init__(self):
            self.trust_env = True
            self.max_redirects = 30
            self.calls = []

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return FakeResponse(url)

        def close(self):
            pass

    class FakeManager:
        sessions = []

        def __init__(self, config):
            self.config = config
            self.session = FakeSession()
            self.sessions.append(self.session)

        def get_session(self):
            return self.session

    model = ChatWebSearchModel()
    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    monkeypatch.setattr(web_download_module, "Manager", FakeManager)
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
        ("knowledge_base:read", "web:search", "web:fetch"),
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
    assert model.calls == 3
    assert "web_search" in model.bound_tool_names
    assert "web_fetch" in model.bound_tool_names
    search_message = model.received_messages[1][-1]
    fetch_message = model.received_messages[2][-1]
    assert isinstance(search_message, ToolMessage)
    assert isinstance(fetch_message, ToolMessage)
    assert search_message.tool_call_id == "chat-web-call"
    assert fetch_message.tool_call_id == "chat-fetch-call"
    assert json.loads(search_message.content)["output"]["results"][0]["url"] == (
        "https://example.com/python-news"
    )
    fetch_output = json.loads(fetch_message.content)["output"]
    assert fetch_output["final_url"] == "https://example.com/python-news"
    assert fetch_output["content_is_untrusted"] is True
    assert model.selected_url == "https://example.com/python-news"
    assert len(ddgs_calls) == 2
    assert len(FakeManager.sessions) == 1
    assert FakeManager.sessions[0].calls[0][0] == "GET"
    assert rewrite_calls == [
        ("今天有哪些重要的 Python 新闻？", [], "trace-chat-web")
    ]
    assert session_writes[0] == (
        "session-chat-web",
        "user",
        "今天有哪些重要的 Python 新闻？",
    )
    assert session_writes[1][0:2] == ("session-chat-web", "assistant")
    assert "Python release news" in session_writes[1][2]
    assert "A new Python release is available" in session_writes[1][2]
    assert "https://example.com/python-news" in session_writes[1][2]
    assert len(metric_writes) == 1
    assert metric_writes[0]["tool_called"] is True
    assert metric_writes[0]["tool_success"] is True
    assert metric_writes[0]["tool_name"] == "web_fetch"
    assert [
        item["tool_name"]
        for item in outcome.result["tool_trace"]
        if item.get("step") == "tool_call" and item.get("handler_executed")
    ] == ["web_search", "web_fetch"]


def test_chat_service_completes_search_and_pdf_fetch_without_rag(
    monkeypatch,
):
    """ChatService 保持原 API，并记录脱敏多工具调用链。"""
    web_search_module = import_module("app.tools.web_search")
    web_download_module = import_module("app.tools.web_download")
    pdf_bytes = (
        Path(__file__).parent / "fixtures" / "web_pdf_text.pdf"
    ).read_bytes()
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
                    "title": "Governed Web PDF Fixture",
                    "href": "https://example.com/document.pdf",
                    "body": "Two page public PDF",
                }
            ]

    class FakeResponse:
        status_code = 200
        encoding = None
        headers = {"Content-Type": "application/pdf"}

        def __init__(self, url):
            self.url = url
            self.closed = False

        def iter_content(self, chunk_size):
            del chunk_size
            yield pdf_bytes

        def close(self):
            self.closed = True

    class FakeSession:
        def __init__(self):
            self.trust_env = True
            self.max_redirects = 30
            self.calls = []
            self.closed = False

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return FakeResponse(url)

        def close(self):
            self.closed = True

    class FakeManager:
        sessions = []

        def __init__(self, config):
            self.config = config
            self.session = FakeSession()
            self.sessions.append(self.session)

        def get_session(self):
            return self.session

    model = ChatWebPdfModel()
    monkeypatch.setattr(web_search_module, "DDGS", FakeDDGS)
    monkeypatch.setattr(web_download_module, "Manager", FakeManager)
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        graph_nodes,
        "run_rag_subgraph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"web PDF must not enter RAG: {args} {kwargs}")
        ),
    )
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
        ("knowledge_base:read", "web:search", "web:fetch"),
    )
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_AUDIT_ENABLED", False)

    session_writes = []
    metric_writes = []
    clock = IncrementingClock()

    def rewrite(question, history, trace_id=""):
        del history, trace_id
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
        now_iso=lambda: "2026-07-23T10:00:00.000",
        perf_counter=clock,
        response_map=CHAT_RESULT_RESPONSE_MAP,
    )
    log_messages = []

    def record_log(message, *args, **kwargs):
        del kwargs
        rendered = message % args if args else message
        log_messages.append(rendered)

    monkeypatch.setattr("app.services.chat_service.logger.info", record_log)

    try:
        outcome = ChatService(dependencies).handle(
            ChatServiceRequest(
                question="读取搜索到的公开 PDF",
                top_k=5,
                decompose_top_k=10,
                session_id="session-chat-pdf",
                trace_id="trace-chat-pdf",
                request_enter_time="2026-07-23T09:59:59.000",
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
    assert model.calls == 3
    assert {"web_search", "web_pdf_fetch"} <= set(model.bound_tool_names)
    search_message = model.received_messages[1][-1]
    pdf_message = model.received_messages[2][-1]
    assert isinstance(search_message, ToolMessage)
    assert isinstance(pdf_message, ToolMessage)
    assert search_message.tool_call_id == "chat-pdf-search-call"
    assert pdf_message.tool_call_id == "chat-pdf-fetch-call"
    pdf_output = json.loads(pdf_message.content)["output"]
    assert pdf_output["page_numbers"] == [1, 2]
    assert "[Page 1]" in pdf_output["content"]
    assert "Governed Web PDF Fixture" in outcome.result["answer"]
    assert model.selected_url in outcome.result["answer"]
    assert session_writes[1][0:2] == ("session-chat-pdf", "assistant")
    assert len(ddgs_calls) == 2
    assert len(FakeManager.sessions) == 1
    assert len(metric_writes) == 1
    assert metric_writes[0]["tool_called"] is True
    assert metric_writes[0]["tool_success"] is True
    assert metric_writes[0]["tool_name"] == "web_pdf_fetch"
    assert [
        item["tool_name"]
        for item in outcome.result["tool_trace"]
        if item.get("step") == "tool_call" and item.get("handler_executed")
    ] == ["web_search", "web_pdf_fetch"]
    chain_logs = [
        message
        for message in log_messages
        if "tool_call_chain=" in message
    ]
    assert len(chain_logs) == 1
    assert "web_search(success," in chain_logs[0]
    assert " -> web_pdf_fetch(success," in chain_logs[0]
    assert "example.com" not in chain_logs[0]
    assert "Governed Web PDF Fixture" not in chain_logs[0]
    assert "Governed PDF page" not in chain_logs[0]
