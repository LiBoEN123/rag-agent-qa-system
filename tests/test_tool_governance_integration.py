"""工具治理结果、直接 BaseTool 入口和 trace 投影集成测试。"""

import json
from copy import deepcopy

import pytest

from app.config import get_settings
from app.agent.tool_events import build_tool_trace, normalize_governance_events
from app.tools.context import ToolInvocationContext
from app.tools.catalog import get_local_tool_specs
from tests.support.local_tools import invoke_local_tool


def test_execution_entry_runs_default_registry_tool(monkeypatch):
    monkeypatch.setattr("app.rag.pipeline.preview_docs", lambda: [])
    result = invoke_local_tool(
        registry=get_local_tool_specs(get_settings()),
        tool_name="list_documents",
        tool_args={},
        context=ToolInvocationContext(trace_id="direct-trace", permissions=("knowledge_base:read",)),
    )
    assert result["ok"] is True
    assert result["governance"]["decision"] == "allow"
    assert result["runtime"]["attempt_count"] == 1
    assert result["tool_call_id"]


def test_public_tool_trace_redacts_args_and_summarizes_chunk_output():
    chunk_text = "private chunk body " * 100
    trace = build_tool_trace(
        step_id=1,
        tool_name="get_chunk_text",
        tool_args={"chunk_id": 1, "password": "raw-password"},
        result={
            "status": "success",
            "ok": True,
            "output": {"chunk": {"text": chunk_text, "token": "raw-token"}},
            "governance": {"decision": "allow", "risk_level": "read_only"},
            "runtime": {"attempt_count": 1},
        },
        cost_ms=0,
    )
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "raw-password" not in serialized
    assert "raw-token" not in serialized
    assert chunk_text not in serialized
    assert trace["sanitized_tool_args"]["password"] == "***REDACTED***"
    assert trace["output_summary"]["mode"] == "none"


def test_knowledge_tool_trace_projects_public_final_evidence():
    chunks = [
        {
            "chunk_id": "final-b",
            "file_name": "b.md",
            "hybrid_score": 0.82,
            "rerank_score": 0.91,
        },
        {
            "chunk_id": "final-c",
            "file_name": "c.md",
            "hybrid_score": 0.73,
            "rerank_score": 0.84,
        },
    ]
    output = {
        "count": 2,
        "chunks": chunks,
        "top_chunk_ids": ["final-b", "final-c"],
        "top_scores": [0.82, 0.73],
    }
    trace = build_tool_trace(
        step_id=1,
        tool_name="search_knowledge_base",
        tool_args={"query": "对比两个政策", "top_k": 2},
        result={"status": "success", "ok": True, "output": output},
        cost_ms=8,
    )

    assert trace["final_chunk_count"] == 2
    assert trace["top_chunk_ids"] == ["final-b", "final-c"]
    assert trace["top_scores"] == [0.82, 0.73]
    assert [hit["chunk_id"] for hit in trace["hits"]] == [
        "final-b",
        "final-c",
    ]


def test_governance_event_normalization_is_pure_and_keeps_separate_step_counts():
    model = {
        "kind": "model", "ok": True, "has_tool_calls": True,
        "requested_tool_call_count": 1, "cost_ms": 2,
        "available_tool_names": ["list_documents"],
    }
    tool = {
        "kind": "tool", "name": "list_documents",
        "args": {},
        "result": {
            "ok": True, "status": "success", "output": {"count": 0},
            "tool_call_id": "list-1", "runtime": {"attempt_count": 1},
        },
        "ok": True, "counted_call": True, "handler_executed": True,
        "cost_ms": 3, "tool_call_id": "list-1",
    }
    blocked = {
        "kind": "tool", "name": "list_documents",
        "result": {"ok": False, "error": "blocked"},
        "counted_call": False, "error_code": "repeated_tool_call",
        "stop_reason": "repeated_tool_call", "tool_call_id": "list-2",
    }
    events = [model, blocked, tool, model, blocked, tool]
    original = deepcopy(events)

    trace = normalize_governance_events(events)

    assert events == original
    assert normalize_governance_events(events) == trace
    assert trace[0] == {
        "step": "agent_iteration", "iteration_index": 1, "status": "success",
        "model_has_tool_calls": True, "requested_tool_call_count": 1,
        "model_cost_ms": 2, "available_tool_names": ["list_documents"],
        "error_type": "",
    }
    assert trace[3] == {**trace[0], "iteration_index": 2}
    assert trace[1] == trace[4] == {
        "step": "tool_call", "action": "tool_call",
        "tool_name": "list_documents", "tool_status": "error", "success": False,
        "error": "blocked", "error_code": "repeated_tool_call",
        "tool_call_id": "list-2", "cost_ms": 0, "handler_executed": False,
        "will_return_to_agent_model": False, "stop_reason": "repeated_tool_call",
        "call_fingerprint": "",
    }
    assert trace[2] == {
        "step": "tool_call", "step_id": 1, "action": "tool_call",
        "tool_name": "list_documents", "tool_args": {},
        "sanitized_tool_args": {}, "tool_status": "success",
        "success": True, "summary": "文档列表工具执行完成", "error": "", "cost_ms": 3,
        "error_code": "", "tool_call_id": "list-1", "policy_decision": "",
        "risk_level": "", "approval_required": False, "args_fingerprint": "",
        "attempt_count": 1, "max_attempts": 0, "timed_out": False,
        "output_summary": {"count": 0}, "failure_kind": "", "handler_executed": True,
        "will_return_to_agent_model": True, "stop_reason": "", "call_fingerprint": "",
    }
    assert trace[5] == {**trace[2], "step_id": 2}


@pytest.mark.parametrize(
    ("result", "success"),
    [
        ({"status": "success"}, True),
        ({"status": "error"}, False),
        ({"ok": False, "status": "success"}, False),
        ({"ok": True, "status": "error"}, True),
        ({"ok": None, "status": "success"}, False),
        ({"ok": "yes"}, True),
        ({}, False),
    ],
)
def test_public_tool_trace_keeps_ok_precedence(result, success):
    trace = build_tool_trace(
        step_id=1, tool_name="list_documents", tool_args={}, result=result, cost_ms=0,
    )
    assert trace["success"] is success
    assert trace["summary"] == ("文档列表工具执行完成" if success else "工具执行失败")
    assert trace["error"] == ("" if success else "Tool execution failed")


@pytest.mark.parametrize(
    ("error_code", "raw_error", "expected"),
    [
        *[(code, "安全策略错误 token=raw-token", "安全策略错误 token=***REDACTED***")
          for code in (
              "permission_denied", "approval_required",
              "missing_idempotency_key", "invalid_invocation_context", "policy_denied",
              "audit_write_failed",
          )],
        ("execution_error", "private backend detail", "工具执行失败（execution_error）"),
        ("", "fallback password=raw-password", "fallback password=***REDACTED***"),
        ("", "", "Tool execution failed"),
        ("policy_denied", "", "Tool execution failed"),
    ],
)
def test_public_tool_trace_keeps_safe_error_policy(error_code, raw_error, expected):
    trace = build_tool_trace(
        step_id=1, tool_name="list_documents", tool_args={}, cost_ms=0,
        result={"ok": False, "error_code": error_code, "error": raw_error},
    )
    assert trace["error"] == expected
    assert trace["summary"] == "工具执行失败"


@pytest.mark.parametrize("tool_name", ["list_documents", "search_knowledge_base", "mcp_probe"])
@pytest.mark.parametrize("output", [None, ["private raw output"], "private raw output"])
def test_public_tool_trace_normalizes_non_dict_output(tool_name, output):
    trace = build_tool_trace(
        step_id=1, tool_name=tool_name, tool_args={}, cost_ms=0,
        result={"ok": True, "output": output},
    )
    assert "private raw output" not in json.dumps(trace)
    if tool_name == "search_knowledge_base":
        assert trace["final_chunk_count"] == 0
        assert trace["top_chunk_ids"] == trace["top_scores"] == trace["hits"] == []
    elif tool_name == "list_documents":
        assert trace["output_summary"] == {"count": 0}
    else:
        assert trace["output_summary"] == {}


@pytest.mark.parametrize("tool_name", ["web_search", "web_fetch", "web_pdf_fetch"])
def test_public_tool_trace_keeps_business_body_out_of_output_summary(tool_name):
    trace = build_tool_trace(
        step_id=1, tool_name=tool_name, tool_args={"password": "raw-password"}, cost_ms=1,
        result={
            "ok": True, "summary": "完成 token=raw-token",
            "output": {
                "text": "private business body", "rows": [{"value": "private row"}],
                "token": "raw-token", "credential": "raw-credential",
            },
        },
    )
    serialized = json.dumps(trace)
    for value in ("raw-password", "raw-token", "raw-credential", "private business body", "private row"):
        assert value not in serialized
    assert "output" not in trace
    assert trace["summary"] == "完成 token=***REDACTED***"


def test_public_nonlocal_tool_trace_preserves_existing_redaction_boundary():
    trace = build_tool_trace(
        step_id=1, tool_name="mcp_probe", tool_args={"token": "raw-token"}, cost_ms=1,
        result={
            "ok": True, "summary": "完成 password=raw-password",
            "output": {"count": 2, "token": "raw-token", "credential": "raw-credential"},
        },
    )
    assert trace["output_summary"] == {
        "count": 2, "token": "***REDACTED***", "credential": "***REDACTED***",
    }
    assert "output" not in trace
    assert trace["summary"] == "完成 password=***REDACTED***"
    assert "raw-" not in json.dumps(trace)


def test_knowledge_tool_trace_caps_redacts_and_keeps_metadata_score_fallbacks():
    chunks = [
        {
            "metadata": {"chunk_id": f"chunk-{index}", "file_name": "token=raw-token"},
            "text": "private full chunk", "rerank_score": 0.9, "hybrid_score": 0.8,
            "bm25_score": 0.7, "vector_score": 0.6,
        }
        for index in range(12)
    ]
    chunks[1].update(chunk_id="explicit-id", file_name="explicit.md")
    chunks[2]["metadata"] = "invalid metadata"
    chunks[3] = "invalid chunk"
    trace = build_tool_trace(
        step_id=1, tool_name="search_knowledge_base", tool_args={}, cost_ms=1,
        result={"ok": True, "output": {
            "count": 12, "chunks": chunks,
            "top_chunk_ids": tuple(f"chunk-{index}" for index in range(12)),
            "top_scores": [0.8] * 12,
        }},
    )
    assert trace["final_chunk_count"] == 12
    assert trace["top_chunk_ids"] == [f"chunk-{index}" for index in range(10)]
    assert trace["top_scores"] == [0.8] * 10
    assert len(trace["hits"]) == 9
    assert trace["hits"][0] == {
        "chunk_id": "chunk-0", "file_name": "token=***REDACTED***",
        "rerank": 0.9, "score": 0.8, "bm25": 0.7, "vector": 0.6,
    }
    assert trace["hits"][1]["chunk_id"] == "explicit-id"
    assert trace["hits"][1]["file_name"] == "explicit.md"
    assert trace["hits"][2]["chunk_id"] is None
    assert trace["hits"][2]["file_name"] is None
    assert trace["hits"][-1]["chunk_id"] == "chunk-9"
    assert "raw-token" not in json.dumps(trace)
    assert "private full chunk" not in json.dumps(trace)
