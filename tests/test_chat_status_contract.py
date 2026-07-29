"""单一 ChatStatusPolicy 与 API 结果的一致性合同。"""

from pathlib import Path

import pytest

from app.agent.results import create_agent_result
from app.api.responses import build_chat_response_from_outcome
from app.api.schemas import ChatData, SearchData
from app.contracts.chat import (
    CHAT_STATUS_POLICIES,
    AgentResultStatus,
    get_chat_status_policy,
)
from app.services.chat_models import ChatServiceOutcome
from app.services.chat_service import _apply_status_policy


REFERENCE = {
    "reference_id": "Reference 1",
    "file_name": "policy.md",
    "source_path": "data/policy.md",
    "chunk_id": "chunk-1",
    "position": 1,
    "snippet": "测试引用",
    "score": 0.9,
    "rerank_score": 0.8,
}

EXPECTED = {
    AgentResultStatus.OK: (200, 0, "success", "ok", "success"),
    AgentResultStatus.NEED_CLARIFICATION: (
        200,
        "NEED_CLARIFICATION",
        "问题信息不足，需要补充后再回答",
        "clarify",
        "clarify",
    ),
    AgentResultStatus.NO_RELEVANT_INFO: (
        200,
        "NO_RELEVANT_INFO",
        "我没有在文档中找到相关信息",
        "no_evidence",
        "no_relevant_info",
    ),
    AgentResultStatus.TOOL_ERROR: (
        200,
        "TOOL_ERROR",
        "工具调用失败",
        "error",
        "tool_error",
    ),
    AgentResultStatus.MODEL_ERROR: (
        200,
        "MODEL_ERROR",
        "模型调用失败",
        "error",
        "model_error",
    ),
    AgentResultStatus.INTERNAL_ERROR: (
        500,
        "INTERNAL_ERROR",
        "服务器内部错误",
        "error",
        "internal_error",
    ),
}


def test_status_policy_is_complete_and_has_no_extra_states():
    assert set(CHAT_STATUS_POLICIES) == set(AgentResultStatus)
    assert {
        policy.result_status for policy in CHAT_STATUS_POLICIES.values()
    } == set(AgentResultStatus)


@pytest.mark.parametrize("status", list(AgentResultStatus))
def test_policy_drives_http_api_and_metric_status(status):
    policy = get_chat_status_policy(status)
    raw = create_agent_result(
        status=status,
        answer="测试回答",
        references=[REFERENCE],
        need_clarification=status is AgentResultStatus.NEED_CLARIFICATION,
        clarification_question=(
            "请补充具体对象"
            if status is AgentResultStatus.NEED_CLARIFICATION
            else ""
        ),
        reason="测试原因",
    )
    resolved = _apply_status_policy(raw, policy)
    outcome = ChatServiceOutcome(
        http_status=policy.http_status,
        code=policy.code,
        message=policy.message,
        question="测试问题",
        result=resolved,
        response_status=policy.public_status,
    )

    response = build_chat_response_from_outcome(
        outcome,
        trace_id="trace-policy",
    )
    payload = response.model_dump()

    assert (
        outcome.http_status,
        payload["code"],
        payload["message"],
        payload["data"]["status"],
        policy.metric_status,
    ) == EXPECTED[status]
    assert payload["data"]["reason"] == "测试原因"
    if status is AgentResultStatus.NEED_CLARIFICATION:
        assert payload["data"]["answer"] == ""
        assert payload["data"]["references"] == []
        assert payload["data"]["need_clarification"] is True
        assert payload["data"]["clarification_question"] == "请补充具体对象"
    elif status in {
        AgentResultStatus.TOOL_ERROR,
        AgentResultStatus.MODEL_ERROR,
        AgentResultStatus.INTERNAL_ERROR,
    }:
        assert payload["data"]["references"] == []
        assert payload["data"]["need_clarification"] is False
        assert payload["data"]["clarification_question"] == ""
    else:
        assert len(payload["data"]["references"]) == 1


def test_unknown_status_fails_closed_to_internal_error_policy():
    policy = get_chat_status_policy("UNKNOWN")

    assert policy is CHAT_STATUS_POLICIES[AgentResultStatus.INTERNAL_ERROR]
    assert policy.http_status == 500
    assert policy.code == "INTERNAL_ERROR"
    assert policy.public_status == "error"
    assert policy.metric_status == "internal_error"


def test_api_response_adapter_does_not_depend_on_runtime_or_graph():
    source = Path("app/api/responses.py").read_text(encoding="utf-8")

    assert "app.agent.runtime" not in source
    assert "app.graph" not in source
    assert "CHAT_RESULT_RESPONSE_MAP" not in source
    assert "ChatResponse" + "Builder" not in source


def test_public_status_openapi_literals_remain_compatible():
    expected = [
        "ok",
        "no_evidence",
        "empty_index",
        "error",
        "clarify",
        "out_of_scope",
    ]

    assert ChatData.model_json_schema()["properties"]["status"]["enum"] == expected
    assert SearchData.model_json_schema()["properties"]["status"]["enum"] == expected
