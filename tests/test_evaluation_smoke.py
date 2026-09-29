"""三个离线集成 case：真实 ChatService/本地文档列表，以及 fixture RAG 预测与报告。"""

from dataclasses import replace
import json
import asyncio
from unittest.mock import AsyncMock

import pytest

from scripts.evaluation import agent, rag_evaluator, rag_predictions, report
from scripts.evaluation.contracts import AgentEvaluationCase


@pytest.mark.parametrize("tool", ["", "find_documents"], ids=["direct", "local_tool"])
def test_minimal_agent_smoke(tool, settings_factory, monkeypatch, tmp_path):
    from app.agent import framework_tools, streaming
    from app.agent.streaming import AgentRuntimeStreamEvent
    from app.agent.results import create_agent_result
    from app.contracts.chat import AgentResultStatus
    from app.services import chat_service
    from tests.support.local_tools import NoOpAuditSink
    from app.tools.catalog import get_local_tool_specs
    from app.tools.policy import ToolPolicyEngine

    settings = settings_factory(mcp_tools_enabled=False)
    monkeypatch.setattr("app.rag.runtime.chunks", [])
    monkeypatch.setattr("app.rag.runtime.RAG_READY", True)
    base = chat_service.get_default_chat_service_dependencies()
    monkeypatch.setattr(chat_service, "get_default_chat_service_dependencies", lambda: replace(
        base, rewrite_query_with_history=lambda question, history, trace_id: {
            "original_question": question, "rewritten_question": question, "rewritten": False,
        },
    ))

    async def fake_runtime(**kwargs):
        calls = []
        if tool:
            spec = next(s for s in get_local_tool_specs(settings) if s.name == tool)
            output = await framework_tools.execute_project_tool(
                spec=spec, tool_args={}, context=kwargs["tool_context"],
                settings=settings, policy_engine=ToolPolicyEngine(), audit_sink=NoOpAuditSink(),
            )
            assert output["ok"] and output["output"] == {"query": None, "count": 0, "total": 0, "documents": []}
            calls.append({"step": "tool_call", "tool_name": tool, "tool_status": "success"})
        yield AgentRuntimeStreamEvent("result", {"result": create_agent_result(
            status=AgentResultStatus.OK, answer="文档列表为空" if tool else "你好", action="agent",
            tool_trace=calls,
        )})

    monkeypatch.setattr(streaming, "stream_agent", fake_runtime)
    case = AgentEvaluationCase.model_validate({
        "id": tool or "direct", "version": 2, "suite": "core", "category": "smoke",
        "question": "请列出知识库文档" if tool else "你好", "expected": {
            "action": "agent", "status": "OK", "tools_contains": [tool] if tool else [],
            "tool_output_keys": ["count", "documents"] if tool else [],
        },
    })
    row = agent.execute_agent_case(case, settings=settings, )
    assert row["passed"] and row["actual_tools"] == ([tool] if tool else [])
    assert row["tool_output_schema_pass"]
    report.write_jsonl(tmp_path / "per_case.jsonl", [row])
    assert json.loads((tmp_path / "per_case.jsonl").read_text()) == row
    assert not settings.sqlite_db_path.exists()


def test_minimal_rag_smoke_judge_disabled(monkeypatch, tmp_path):
    from scripts.evaluation import deepeval_adapter as judge

    def deny(*args, **kwargs):
        pytest.fail("judge must remain disabled")

    monkeypatch.setattr(judge, "get_judge_model", deny)
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **kwargs: None)
    chunk = {"chunk_id": 1, "text": "评测规则", "metadata": {"source_file": "rules.md"}}
    from app.services.chat_models import ChatServiceOutcome
    from app.rag import runtime as pipeline
    monkeypatch.setattr(pipeline, "chunks", [chunk])
    def chat_turn(*args, **kwargs):
        from app.agent import framework_tools
        from types import SimpleNamespace
        asyncio.run(framework_tools.execute_project_tool(spec=SimpleNamespace(name="search_knowledge_base"), tool_args={}))
        return ChatServiceOutcome(http_status=200, code=200, message="ok", question="规则是什么", result={
            "answer": "评测规则", "status": "OK", "references": [{"chunk_id": 1}],
            "tool_trace": [{"step": "tool_call", "tool_call_id": "a"}], "llm_call_count": 2,
        })
    monkeypatch.setattr("app.agent.framework_tools.execute_project_tool", AsyncMock(return_value={
        "tool_call_id": "a", "ok": True, "output": {"evidence": [
            {"metadata": {"source_type": "knowledge_base", "chunk_id": 1}},
        ]},
    }))
    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", chat_turn)
    eval_file, pred_file, chunks_file = [tmp_path / n for n in ("eval.json", "predictions.csv", "chunks.json")]
    eval_file.write_text(json.dumps([{
        "qid": "q1", "question": "规则是什么", "expected_behavior": "answer",
        "gold_key_points": ["评测规则"], "gold_chunk_ids": [1], "gold_doc_ids": ["rules.md"],
    }]))
    chunks_file.write_text(json.dumps([chunk]))
    rag_predictions.run_predictions(eval_file=eval_file, out_file=pred_file)
    result = rag_evaluator.run_rag_evaluation(
        eval_file=eval_file, pred_file=pred_file, out_dir=tmp_path / "rag", k=1,
        chunks_file=chunks_file, index_source="chunks-file", use_llm_judge=False,
    )
    assert not result["judge_enabled"]
    assert result["summary"]["overall"]["llm_judge_coverage"] == 0
    assert result["rows"][0]["retrieval_gold_chunk_recall_at_k"] == 1
    assert {p.name for p in (tmp_path / "rag").iterdir()} == {
        "per_question_report.csv",
    }
