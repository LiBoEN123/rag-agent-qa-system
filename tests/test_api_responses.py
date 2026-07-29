"""具体 ChatResponse 适配函数的边界测试。"""

from copy import deepcopy

from app.agent.results import create_agent_result
from app.api.responses import build_chat_response_from_outcome
from app.contracts.chat import AgentResultStatus
from app.services.chat_models import ChatServiceOutcome


def test_response_adapter_clamps_confidence_and_keeps_llm_usage():
    result = create_agent_result(
        status=AgentResultStatus.OK,
        answer="回答",
        tool_trace=[
            {
                "step": "query_rewrite",
                "rewrite_llm_used": True,
            },
            {
                "step": "llm_usage",
                "router_llm_used": False,
                "decompose_llm_used": True,
                "answer_llm_used": True,
                "llm_call_count": 2,
            },
        ],
    )
    result["retrieval_confidence"] = 2.5
    result["answer_confidence"] = -1
    outcome = ChatServiceOutcome(
        http_status=200,
        code=0,
        message="success",
        question="问题",
        result=result,
    )

    payload = build_chat_response_from_outcome(
        outcome,
        trace_id="trace-response",
    ).model_dump()

    assert payload["data"]["retrieval_confidence"] == 1.0
    assert payload["data"]["answer_confidence"] == 0.0
    assert payload["data"]["router_llm_used"] is False
    assert payload["data"]["decompose_llm_used"] is True
    assert payload["data"]["answer_llm_used"] is True
    assert payload["data"]["llm_call_count"] == 3


def test_response_adapter_does_not_mutate_outcome_result():
    result = create_agent_result(
        status=AgentResultStatus.NEED_CLARIFICATION,
        clarification_question="请补充对象",
        need_clarification=True,
        tool_trace=[{"step": "query_rewrite", "rewrite_llm_used": False}],
    )
    original = deepcopy(result)
    outcome = ChatServiceOutcome(
        http_status=200,
        code="NEED_CLARIFICATION",
        message="问题信息不足，需要补充后再回答",
        question="问题",
        result=result,
        response_status="clarify",
    )

    build_chat_response_from_outcome(
        outcome,
        trace_id="trace-response",
    )

    assert result == original
