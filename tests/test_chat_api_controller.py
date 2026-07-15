import inspect

import app.api.chat as chat_api
from app.services.chat_service import ChatServiceOutcome, ChatServiceRequest


class DummyState:
    trace_id = "trace-from-middleware"
    request_enter_time = "2026-07-14T10:00:00.000"
    request_enter_perf = 123.5


class DummyRequest:
    state = DummyState()


class FakeChatService:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def handle(self, request):
        self.requests.append(request)
        return self.outcome


def test_chat_controller_only_adapts_http_context(monkeypatch):
    outcome = ChatServiceOutcome(
        http_status=503,
        code="NO_DOCUMENTS",
        message="知识库未准备完成",
        question="原始问题",
        result={
            "answer": "",
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "知识库未准备完成",
            "action": "search_knowledge_base",
            "tool_result_summary": "",
            "tool_trace": [],
        },
        response_status="empty_index",
    )
    service = FakeChatService(outcome)
    monkeypatch.setattr(chat_api, "chat_service", service)
    response = chat_api.Response()

    api_result = chat_api.ask_api(
        chat_api.ChatRequest(
            question="原始问题",
            top_k=4,
            decompose_top_k=8,
            session_id="session-1",
        ),
        DummyRequest(),
        response,
    )

    assert len(service.requests) == 1
    assert service.requests[0] == ChatServiceRequest(
        question="原始问题",
        top_k=4,
        decompose_top_k=8,
        session_id="session-1",
        trace_id="trace-from-middleware",
        request_enter_time="2026-07-14T10:00:00.000",
        request_enter_perf=123.5,
    )
    assert response.status_code == 503
    assert api_result.model_dump() == {
        "code": "NO_DOCUMENTS",
        "message": "知识库未准备完成",
        "trace_id": "trace-from-middleware",
        "data": {
            "question": "原始问题",
            "answer": "",
            "references": [],
            "retrieval_confidence": 0.0,
            "answer_confidence": 0.0,
            "need_clarification": False,
            "status": "empty_index",
            "clarification_question": "",
            "reason": "知识库未准备完成",
            "action": "search_knowledge_base",
            "tool_result_summary": "",
            "tool_trace": [],
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
        },
    }


def test_chat_controller_contains_no_business_orchestration_calls():
    source = inspect.getsource(chat_api.ask_api)

    assert "route_agent_action" not in source
    assert "rewrite_query_with_history" not in source
    assert "run_agent_langgraph" not in source
    assert "get_history" not in source
    assert "append_message" not in source
    assert "collect_trace_metrics" not in source
