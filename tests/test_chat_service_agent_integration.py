"""ChatService 到真实 Agent Graph、Provider 和本地工具的主路径测试。"""

from pydantic import RootModel

import asyncio
import json
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.agent import runtime as agent_runtime
from app.agent.runtime import reset_agent_runtime
from app.contracts.chat import ToolMode
from app.rag import graph as rag_graph
from app.rag import pipeline as rag_pipeline
from app.services.chat_service import (
    ChatService,
    ChatServiceDependencies,
)
from app.services.chat_models import ChatServiceRequest
from app.storage.database import clear_storage_engine_cache
from app.storage.memory import append_message_sqlite, get_history_sqlite
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools
from tests.support.agent_runtime import FakeToolCallingModel
from app.agent.final_response import FinalAgentResponse
from app.tools.result import tool_success


def collect_outcome(service, request):
    async def collect():
        events = [event async for event in service.stream(request)]
        assert events[-1].event == "done"
        return events[-2].data["outcome"]

    return asyncio.run(collect())


class ChatEchoArgs(BaseModel):
    text: str


class ChatAgentModel(FakeToolCallingModel):
    def __init__(self):
        super().__init__([])
        self.calls = 0

    async def ainvoke(self, messages):
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
        return self.final_response(outcome="answer", answer="ChatService 主路径完成。")


class ChatWebSearchModel(ChatAgentModel):
    def __init__(self):
        super().__init__()
        self.selected_url = ""

    async def ainvoke(self, messages):
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
        return self.final_response(
            outcome="answer", evidence_ids=[output["evidence"][0]["id"]],
            answer=(
                f"{output['title']}：{output['evidence'][0]['page_content']} "
                f"来源：{output['final_url']}"
            )
        )


class ChatWebPdfModel(ChatAgentModel):
    """在 ChatService 主路径中搜索 PDF、读取页范围并回答。"""

    def __init__(self):
        super().__init__()
        self.selected_url = ""

    async def ainvoke(self, messages):
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
        return self.final_response(
            outcome="answer", evidence_ids=[output["evidence"][0]["id"]],
            answer=(
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


def test_chat_service_runs_real_create_agent_provider_and_local_tool(
    monkeypatch,
    settings_env,
):
    """ChatService 通过真实 create_agent、Provider 与本地工具完成调用。"""
    reset_agent_runtime()
    registry = []
    handler_contexts: list[ToolInvocationContext] = []
    handler_args: list[dict] = []
    routed_questions: list[str] = []

    def handler(args, context):
        handler_args.append(dict(args))
        handler_contexts.append(context)
        return tool_success("chat_echo", args, {"text": args["text"]})

    registry.append(
        ToolSpec(
            name="chat_echo",
            version="1.0.0",
            description="回显 ChatService 集成测试文本",
            output_model=RootModel[dict],
            handler=handler,
            args_model=ChatEchoArgs,
        )
    )
    model = ChatAgentModel()
    original_route = agent_runtime.route_agent_action

    def recording_route(question):
        routed_questions.append(question)
        return original_route(question)

    monkeypatch.setattr(agent_runtime, "route_agent_action", recording_route)
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_local_tools",
        lambda settings: local_tools(registry),
    )
    settings = settings_env(
        agent_max_iterations=6,
        agent_max_tool_calls=10,
        agent_max_tool_errors=3,
        agent_max_consecutive_same_tool_calls=2,
        agent_max_consecutive_same_tool_errors=2,
        agent_max_total_cost_ms=120000,
        tool_default_principal_id="chat-user",
        tool_default_permissions=("knowledge_base:read",),
        tool_audit_enabled=False,
    )

    session_writes: list[tuple[str, str, str]] = []
    metric_writes: list[dict] = []
    bad_cases: list[dict] = []
    clock = IncrementingClock()

    def persist_message(session_id, role, content):
        session_writes.append((session_id, role, content))
        append_message_sqlite(session_id, role, content)

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
        settings_provider=lambda: settings,
        get_history=get_history_sqlite,
        rewrite_query_with_history=rewrite,
        log_and_write_metrics=lambda record: metric_writes.append(dict(record)),
        write_bad_cases=lambda record: bad_cases.append(dict(record)),
        append_message=persist_message,
        now_iso=lambda: "2026-07-20T10:00:00.000",
        perf_counter=clock,
    )
    service = ChatService(dependencies)

    try:
        outcome = collect_outcome(
            service,
            ChatServiceRequest(
                question="请走完整 Agent 工具主路径",
                session_id="session-chat-integration",
                trace_id="trace-chat-integration",
                request_enter_time="2026-07-20T09:59:59.000",
                request_enter_perf=99.0,
            )
        )
    finally:
        reset_agent_runtime()

    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.result["status"] == "OK"
    assert outcome.result["answer"] == "ChatService 主路径完成。"
    assert outcome.question == "请走完整 Agent 工具主路径"
    assert routed_questions == ["请执行改写后的完整 Agent 工具主路径"]
    assert model.calls == 2
    assert model.received_messages[0][-1].content == (
        "请执行改写后的完整 Agent 工具主路径"
    )
    assert model.bound_tool_names == ["chat_echo", model.structured_tool_name]
    assert isinstance(model.received_messages[1][-1], ToolMessage)
    tool_payload = json.loads(model.received_messages[1][-1].content)
    assert tool_payload["output"] == {"text": "完整主路径"}
    assert handler_args == [{"text": "完整主路径"}]
    assert len(handler_contexts) == 1
    context = handler_contexts[0]
    assert context.trace_id == "trace-chat-integration"
    assert context.session_id == "session-chat-integration"
    assert context.principal_id == "chat-user"
    assert context.permissions == ("knowledge_base:read",)
    assert context.source == "chat_api"
    assert context.tool_mode is ToolMode.FULL
    assert session_writes == [
        ("session-chat-integration", "user", "请走完整 Agent 工具主路径"),
        (
            "session-chat-integration",
            "assistant",
            "ChatService 主路径完成。",
        ),
    ]
    clear_storage_engine_cache()
    assert get_history_sqlite("session-chat-integration") == [
        {"role": "user", "content": "请走完整 Agent 工具主路径"},
        {"role": "assistant", "content": "ChatService 主路径完成。"},
    ]
    tool_calls = [
        step for step in outcome.result["tool_trace"] if step.get("step") == "tool_call"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "chat_echo"
    assert tool_calls[0]["handler_executed"] is True
    assert {step["step"] for step in outcome.result["tool_trace"]} >= {
        "query_rewrite", "decide", "tool_call", "llm_usage", "compose_final_answer",
    }
    assert outcome.result["references"] == []
    assert len(metric_writes) == 1
    decide_trace = next(
        step
        for step in outcome.result["tool_trace"]
        if step.get("step") == "decide"
    )
    assert metric_writes[0]["route_cost_ms"] == decide_trace["route_cost_ms"]
    assert bad_cases == []


@pytest.mark.parametrize("tool_mode", [ToolMode.RAG_ONLY, ToolMode.FULL])
def test_chat_service_returns_kb_evidence_to_agent_with_request_tool_mode(monkeypatch, settings_env, tool_mode):
    settings = settings_env(tool_default_permissions=("knowledge_base:read", "web:search"))
    model = FakeToolCallingModel([
        AIMessage(content="", tool_calls=[{
            "name": "search_knowledge_base", "args": {"query": "学习年限", "top_k": 8},
            "id": "kb-mode-call", "type": "tool_call",
        }]),
        FinalAgentResponse(outcome="no_evidence", answer="无法根据当前证据确定学习年限。"),
    ])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    search, rerank = Mock(return_value=[]), Mock(return_value=[])
    rag = Mock(side_effect=AssertionError("Agent KB path must not call legacy RAG"))
    monkeypatch.setattr(rag_pipeline, "RAG_READY", True)
    monkeypatch.setattr(rag_pipeline, "search_relevant_chunks", search)
    monkeypatch.setattr(rag_pipeline, "rerank", rerank)
    monkeypatch.setattr(rag_graph, "run_rag_subgraph", rag)
    service = ChatService(ChatServiceDependencies(
        settings_provider=lambda: settings,
        get_history=lambda session: [],
        rewrite_query_with_history=lambda **kwargs: {
            "rewritten_question": "本科生学习年限是多少？", "rewrite_llm_used": True,
        },
        log_and_write_metrics=lambda record: None,
        write_bad_cases=lambda record: None,
        append_message=lambda *args: None,
        now_iso=lambda: "2026-09-16T10:00:00.000",
        perf_counter=IncrementingClock(),
    ))
    outcome = collect_outcome(service, ChatServiceRequest(
        question="学习年限是多少？",
        session_id="kb-mode-session", trace_id="kb-mode-trace", tool_mode=tool_mode,
    ))
    assert outcome.result["status"] == "NO_RELEVANT_INFO"
    assert outcome.result["answer"] == "无法根据当前证据确定学习年限。"
    assert outcome.result["references"] == []
    assert outcome.result["llm_call_count"] == 3  # 两轮 Agent + 一次 Query Rewrite
    assert len(model.received_messages) == 2
    first_messages = model.received_messages[0]
    assert first_messages[-1].content == "本科生学习年限是多少？"
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage) and message.tool_call_id == "kb-mode-call"
    assert json.loads(message.content)["output"]["count"] == 0
    search.assert_called_once_with("学习年限", top_k=8)
    rerank.assert_called_once_with([], "学习年限")
    rag.assert_not_called()
    if tool_mode is ToolMode.RAG_ONLY:
        assert model.bound_tool_names == ["search_knowledge_base", model.structured_tool_name]
    else:
        assert {"search_knowledge_base", "web_search", "compare_documents"} <= set(model.bound_tool_names)


def test_chat_service_completes_web_search_and_fetch_through_production_chain(
    monkeypatch,
    settings_env,
):
    """ChatService 通过生产调用链完成 Web 搜索与抓取。"""
    web_search_module = import_module("app.tools.web_search")
    web_download_module = import_module("app.tools.web_download")
    reset_agent_runtime()
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
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    settings = settings_env(
        agent_max_iterations=6,
        agent_max_tool_calls=10,
        agent_max_tool_errors=3,
        agent_max_consecutive_same_tool_calls=2,
        agent_max_consecutive_same_tool_errors=2,
        agent_max_total_cost_ms=120000,
        tool_default_principal_id="web-user",
        tool_default_permissions=(
            "knowledge_base:read",
            "web:search",
            "web:fetch",
        ),
        tool_audit_enabled=False,
    )

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
        settings_provider=lambda: settings,
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
        log_and_write_metrics=lambda record: metric_writes.append(dict(record)),
        write_bad_cases=lambda record: None,
        append_message=lambda session_id, role, content: session_writes.append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-07-22T10:00:00.000",
        perf_counter=clock,
    )

    try:
        outcome = collect_outcome(
            ChatService(dependencies),
            ChatServiceRequest(
                question="今天有哪些重要的 Python 新闻？",
                session_id="session-chat-web",
                trace_id="trace-chat-web",
                request_enter_time="2026-07-22T09:59:59.000",
                request_enter_perf=99.0,
            )
        )
    finally:
        reset_agent_runtime()

    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.result["status"] == "OK"
    assert len(outcome.result["references"]) == 1
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
    settings_env,
):
    """ChatService 保持原 API，并记录脱敏多工具调用链。"""
    web_search_module = import_module("app.tools.web_search")
    web_download_module = import_module("app.tools.web_download")
    pdf_bytes = (
        Path(__file__).parent / "fixtures" / "web_pdf_text.pdf"
    ).read_bytes()
    reset_agent_runtime()
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
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        rag_graph,
        "run_rag_subgraph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"web PDF must not enter RAG: {args} {kwargs}")
        ),
    )
    settings = settings_env(
        agent_max_iterations=6,
        agent_max_tool_calls=10,
        agent_max_tool_errors=3,
        agent_max_consecutive_same_tool_calls=2,
        agent_max_consecutive_same_tool_errors=2,
        agent_max_total_cost_ms=120000,
        tool_default_principal_id="web-user",
        tool_default_permissions=(
            "knowledge_base:read",
            "web:search",
            "web:fetch",
        ),
        tool_audit_enabled=False,
    )

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
        settings_provider=lambda: settings,
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
        log_and_write_metrics=lambda record: metric_writes.append(dict(record)),
        write_bad_cases=lambda record: None,
        append_message=lambda session_id, role, content: session_writes.append(
            (session_id, role, content)
        ),
        now_iso=lambda: "2026-07-23T10:00:00.000",
        perf_counter=clock,
    )
    log_messages = []

    def record_log(message, *args, **kwargs):
        del kwargs
        rendered = message % args if args else message
        log_messages.append(rendered)

    monkeypatch.setattr("app.services.chat_service.logger.info", record_log)

    try:
        outcome = collect_outcome(
            ChatService(dependencies),
            ChatServiceRequest(
                question="读取搜索到的公开 PDF",
                session_id="session-chat-pdf",
                trace_id="trace-chat-pdf",
                request_enter_time="2026-07-23T09:59:59.000",
                request_enter_perf=99.0,
            )
        )
    finally:
        reset_agent_runtime()

    assert outcome.http_status == 200
    assert outcome.code == 0
    assert outcome.result["status"] == "OK"
    assert len(outcome.result["references"]) == 1
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
    assert "[Page 1]" in pdf_output["evidence"][0]["page_content"]
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
