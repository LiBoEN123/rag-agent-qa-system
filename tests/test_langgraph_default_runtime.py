"""唯一 LangGraph API 链路的单次调用与兼容协议回归。"""

from dataclasses import replace

import app.api.chat as chat_api
from app.services.chat_service import (
    ChatService,
    get_default_chat_service_dependencies,
)


class _RequestState:
    trace_id = "trace-default-langgraph"
    request_enter_time = "2026-07-14T12:00:00.000"
    request_enter_perf = 10.0


class _Request:
    state = _RequestState()


def test_default_langgraph_api_path_executes_each_boundary_once(monkeypatch):
    calls = {
        "rewrite": 0,
        "route": 0,
        "rag_ready": 0,
        "langgraph": 0,
        "session_turn": 0,
        "messages": [],
        "metrics": 0,
    }
    decision = {
        "decision": "use_knowledge_base",
        "action": "search_knowledge_base",
        "tool_name": "search_knowledge_base",
        "tool_args": {"query": "默认问题", "top_k": 5},
        "reason": "知识库问题",
        "router_source": "rule_based",
        "router_llm_used": False,
    }

    def rewrite(question, history, trace_id=""):
        calls["rewrite"] += 1
        return {
            "original_question": question,
            "rewritten_question": question,
            "history_used": False,
            "rewritten": False,
            "need_check": False,
            "need_rewrite": False,
            "rewrite_llm_used": False,
            "reason": "无需改写",
            "error": "",
        }

    def route(question, top_k, trace_id=""):
        calls["route"] += 1
        return decision

    def rag_status():
        calls["rag_ready"] += 1
        return {"ready": True, "message": "OK"}

    def langgraph(**kwargs):
        calls["langgraph"] += 1
        assert kwargs["decision"] is decision
        return {
            "status": "OK",
            "answer": "LangGraph 默认回答",
            "references": [
                {
                    "reference_id": "Reference 1",
                    "file_name": "policy.md",
                    "source_path": "/kb/policy.md",
                    "chunk_id": 1,
                    "position": 1,
                    "snippet": "默认路径证据",
                    "score": 0.9,
                    "rerank_score": 0.8,
                }
            ],
            "retrieval_confidence": 0.9,
            "answer_confidence": 0.8,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "证据充分",
            "action": "search_knowledge_base",
            "tool_result_summary": "知识库检索完成，召回 1 个 chunk",
            "tool_trace": [
                {
                    "step": "tool_call",
                    "action": "tool_call",
                    "tool_name": "search_knowledge_base",
                    "success": True,
                    "tool_status": "success",
                },
                {
                    "step": "compose_final_answer",
                    "mode": "rag",
                    "status": "ok",
                },
                {
                    "step": "llm_usage",
                    "router_llm_used": False,
                    "decompose_llm_used": False,
                    "answer_llm_used": True,
                    "llm_call_count": 1,
                },
            ],
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": True,
            "llm_call_count": 1,
        }

    dependencies = replace(
        get_default_chat_service_dependencies(),
        get_history=lambda session_id: [],
        rewrite_query_with_history=rewrite,
        route_agent_action=route,
        get_rag_status=rag_status,
        run_agent_langgraph=langgraph,
        append_message=lambda session_id, role, content: calls[
            "messages"
        ].append((session_id, role, content)),
        log_and_write_metrics=lambda record: calls.__setitem__(
            "metrics", calls["metrics"] + 1
        ),
    )
    service = ChatService(dependencies)
    original_save_session_turn = service._save_session_turn

    def save_session_turn(context):
        calls["session_turn"] += 1
        original_save_session_turn(context)

    monkeypatch.setattr(service, "_save_session_turn", save_session_turn)
    monkeypatch.setattr(chat_api, "chat_service", service)
    response = chat_api.Response()

    result = chat_api.ask_api(
        chat_api.ChatRequest(
            question="默认问题",
            top_k=5,
            session_id="session-default",
        ),
        _Request(),
        response,
    ).model_dump()

    assert response.status_code == 200
    assert calls == {
        "rewrite": 1,
        "route": 1,
        "rag_ready": 1,
        "langgraph": 1,
        "session_turn": 1,
        "messages": [
            ("session-default", "user", "默认问题"),
            ("session-default", "assistant", "LangGraph 默认回答"),
        ],
        "metrics": 1,
    }
    assert result["code"] == 0
    assert result["data"]["status"] == "ok"
    assert result["data"]["references"][0]["reference_id"] == "Reference 1"
    assert result["data"]["llm_call_count"] == 1
    assert [
        step["step"] for step in result["data"]["tool_trace"]
    ].count("tool_call") == 1
