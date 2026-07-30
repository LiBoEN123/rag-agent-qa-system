"""唯一 AgentRunResult 模型和跨运行层状态 mapper 合同。"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.agent.results import (
    AgentRunResult,
    build_agent_result,
    build_internal_error_result,
    map_agent_status_to_result_status,
    map_rag_status_to_agent_status,
    validate_agent_result_payload,
)
from app.contracts.chat import AgentResultStatus


def _valid_payload(**updates):
    payload = {
        "status": "OK",
        "answer": "测试回答",
        "references": [{"reference_id": "Reference 1"}],
        "retrieval_confidence": 0.8,
        "answer_confidence": 0.7,
        "need_clarification": False,
        "clarification_question": "",
        "reason": "测试通过",
        "action": "agent",
        "tool_result_summary": "工具摘要",
        "tool_trace": [{"step": "llm_usage"}],
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 1,
        "timings": {"llm_cost_ms": 3},
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    ("status", "need_clarification"),
    [
        (AgentResultStatus.OK, False),
        (AgentResultStatus.NEED_CLARIFICATION, True),
        (AgentResultStatus.INTERNAL_ERROR, False),
    ],
)
def test_agent_run_result_accepts_complete_results(
    status,
    need_clarification,
):
    payload = _valid_payload(
        status=status.value,
        need_clarification=need_clarification,
        clarification_question="请补充信息" if need_clarification else "",
    )

    result = AgentRunResult.model_validate(payload)
    dumped = result.model_dump(mode="json")

    assert dumped["status"] == status.value
    assert isinstance(dumped["status"], str)
    assert dumped["timings"] == {"llm_cost_ms": 3}


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "UNKNOWN"},
        {"unexpected": "field"},
        {"retrieval_confidence": -0.1},
        {"answer_confidence": 1.1},
        {"llm_call_count": -1},
        {"references": ["not-a-dict"]},
        {"tool_trace": ["not-a-dict"]},
        {"router_llm_used": True},
    ],
)
def test_agent_run_result_rejects_invalid_payload(updates):
    with pytest.raises(ValidationError):
        AgentRunResult.model_validate(_valid_payload(**updates))


def test_agent_run_result_requires_every_runtime_field():
    payload = _valid_payload()
    payload.pop("timings")

    with pytest.raises(ValidationError):
        AgentRunResult.model_validate(payload)


@pytest.mark.parametrize(
    ("agent_status", "result_status"),
    [
        ("running", AgentResultStatus.INTERNAL_ERROR),
        ("ok", AgentResultStatus.OK),
        ("clarify", AgentResultStatus.NEED_CLARIFICATION),
        ("direct_answer", AgentResultStatus.OK),
        ("tool_error", AgentResultStatus.TOOL_ERROR),
        ("model_error", AgentResultStatus.MODEL_ERROR),
        ("no_evidence", AgentResultStatus.NO_RELEVANT_INFO),
        ("error", AgentResultStatus.INTERNAL_ERROR),
        ("unknown", AgentResultStatus.INTERNAL_ERROR),
    ],
)
def test_every_agent_status_has_one_result_mapping(
    agent_status,
    result_status,
):
    assert map_agent_status_to_result_status(agent_status) is result_status


@pytest.mark.parametrize(
    ("rag_status", "agent_status"),
    [
        ("OK", "ok"),
        ("NEED_CLARIFICATION", "clarify"),
        ("NO_RELEVANT_INFO", "no_evidence"),
        ("MODEL_ERROR", "model_error"),
        ("MODEL_BAD_OUTPUT", "model_error"),
        ("unexpected", "error"),
    ],
)
def test_rag_status_enters_agent_through_one_mapper(
    rag_status,
    agent_status,
):
    assert map_rag_status_to_agent_status(rag_status) == agent_status


def test_build_agent_result_is_pure_and_returns_valid_model():
    state = {
        "status": "ok",
        "final_answer": "回答",
        "references": [],
        "reason": "completed",
        "decision": {"action": "agent"},
        "observations": [],
        "tool_trace": [{"step": "decide"}],
        "decompose_llm_used": False,
        "answer_llm_used": True,
        "llm_call_count": 1,
    }
    original = deepcopy(state)

    result = build_agent_result(state)

    assert state == original
    assert isinstance(result, AgentRunResult)
    assert result.status is AgentResultStatus.OK
    assert result.model_dump(mode="json")["status"] == "OK"


def test_internal_error_result_has_the_complete_contract_and_is_sanitized():
    result = build_internal_error_result(
        RuntimeError("api_key=secret"),
        trace_step="graph_error",
    )

    assert set(result) == set(AgentRunResult.model_fields)
    assert result["status"] == "INTERNAL_ERROR"
    assert result["timings"] == {}
    assert "secret" not in str(result)
    assert validate_agent_result_payload(result) == result
