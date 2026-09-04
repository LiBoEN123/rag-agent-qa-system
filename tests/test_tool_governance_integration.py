"""工具治理结果、直接 BaseTool 入口和 trace 投影集成测试。"""

import json

from app.agent.state import AgentObservation
from app.agent.tool_events import build_tool_trace
from app.tools.context import ToolInvocationContext
from app.tools.registry import get_tool_registry
from tests.support.local_tools import invoke_local_tool


def test_execution_entry_runs_default_registry_tool():
    result = invoke_local_tool(
        registry=get_tool_registry(),
        tool_name="simple_calculator",
        tool_args={"expression": "1 + 2"},
        context=ToolInvocationContext(trace_id="direct-trace"),
    )
    assert result["ok"] is True
    assert result["governance"]["decision"] == "allow"
    assert result["runtime"]["attempt_count"] == 1
    assert result["tool_call_id"]


def test_public_tool_trace_redacts_args_and_summarizes_chunk_output():
    chunk_text = "private chunk body " * 100
    observation = AgentObservation(
        step_id=1,
        tool_name="get_chunk_text",
        success=True,
        summary="chunk 原文工具执行完成",
        output={"chunk": {"text": chunk_text, "token": "raw-token"}},
        error="",
        cost_ms=0,
    )
    trace = build_tool_trace(
        step_id=1,
        tool_name="get_chunk_text",
        tool_args={"chunk_id": 1, "password": "raw-password"},
        result={
            "status": "success",
            "ok": True,
            "output": observation["output"],
            "governance": {"decision": "allow", "risk_level": "read_only"},
            "runtime": {"attempt_count": 1},
        },
        observation=observation,
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
    observation = AgentObservation(
        step_id=1,
        tool_name="search_knowledge_base",
        success=True,
        summary="检索到 2 个候选 chunk",
        output=output,
        error="",
        cost_ms=8,
    )

    trace = build_tool_trace(
        step_id=1,
        tool_name="search_knowledge_base",
        tool_args={"query": "对比两个政策", "top_k": 2},
        result={"status": "success", "ok": True, "output": output},
        observation=observation,
    )

    assert trace["final_chunk_count"] == 2
    assert trace["top_chunk_ids"] == ["final-b", "final-c"]
    assert trace["top_scores"] == [0.82, 0.73]
    assert [hit["chunk_id"] for hit in trace["hits"]] == [
        "final-b",
        "final-c",
    ]
