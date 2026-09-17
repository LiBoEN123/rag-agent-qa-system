"""具体 ChatResponse 适配函数的边界测试。"""

from copy import deepcopy

import pytest

from app.agent.results import create_agent_result
from app.api.responses import build_chat_response_from_outcome
from app.contracts.chat import AgentResultStatus
from app.services.chat_models import ChatServiceOutcome


@pytest.mark.parametrize("llm_call_count", [0, 3, 7])
def test_response_adapter_clamps_confidence_and_trusts_result_usage(llm_call_count):
    result = create_agent_result(
        status=AgentResultStatus.OK,
        answer="回答",
        answer_llm_used=True,
        llm_call_count=llm_call_count,
        tool_trace=[
            {
                "step": "query_rewrite",
                "rewrite_llm_used": True,
            },
            {
                "step": "llm_usage",
                "router_llm_used": True,
                "answer_llm_used": False,
                "llm_call_count": 2,
            },
        ],
    )
    result["retrieval_confidence"] = 2.5
    result["answer_confidence"] = -1
    original = deepcopy(result)
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
    assert "decompose_llm_used" not in payload["data"]
    assert payload["data"]["answer_llm_used"] is True
    assert payload["data"]["llm_call_count"] == llm_call_count
    assert payload["data"]["tool_trace"] == original["tool_trace"]
    assert result == original


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
