"""Prediction v3 通过真实 ChatService / create_agent / governed tools 的确定性验收。"""

import csv
import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import framework_agent, framework_tools
from app.agent.final_response import FinalAgentResponse
from app.contracts.chat import ToolMode
from app.rag import pipeline
from scripts.evaluation import rag_predictions, runner
from scripts.evaluation.agent import capture_tool_output_keys
from tests.support.agent_runtime import FakeToolCallingModel


def chunk(cid, file="A.md"):
    return {"chunk_id": cid, "file_name": file, "text": f"证据正文 {cid}", "hybrid_score": 0.9, "metadata": {"chunk_id": cid, "file_name": file}}


def call(name="search_knowledge_base", cid="a", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def finish(model, messages):
    docs = [doc for message in messages if isinstance(message, ToolMessage)
            for doc in json.loads(message.content).get("output", {}).get("evidence", [])]
    return model.final_response(outcome="answer", answer="A 与 B 的申请条件见已检索证据。",
                                evidence_ids=[doc["id"] for doc in docs if doc["metadata"]["chunk_id"] in (1, 3)])


@pytest.fixture
def setup_prediction(monkeypatch):
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "chunks", [chunk(1), chunk(2), chunk(3, "B.md")])
    monkeypatch.setattr(pipeline, "rerank", lambda chunks, query: chunks)
    contexts = []
    original = framework_tools.execute_project_tool

    def execute(**kwargs):
        contexts.append(kwargs["context"])
        return original(**kwargs)

    monkeypatch.setattr(framework_tools, "execute_project_tool", execute)
    monkeypatch.setattr(framework_tools, "list_mcp_tools_for_context", Mock(side_effect=AssertionError("MCP discovery forbidden")))
    return contexts


def run(monkeypatch, responses, question="比较 A 和 B 的申请条件、材料及办理时间"):
    model = FakeToolCallingModel(responses)
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    result, trace = rag_predictions.run_rag_once_with_result(question)
    return result, trace, model


@pytest.mark.parametrize("category", ["cross_doc", "partial_answer"])
def test_multi_subquestion_observes_a_then_changes_query_and_top_k(monkeypatch, setup_prediction, category):
    dataset = json.loads((Path(__file__).resolve().parents[1] / "data/eval/rag_eval_questions_v2_semantic_fixed.json").read_text())
    question = next(q["question"] for q in dataset["items"] if q["category"] == category and q["difficulty"] == "hard")
    search = Mock(side_effect=[[chunk(1), chunk(2)], [chunk(2), chunk(3, "B.md")]])
    monkeypatch.setattr(pipeline, "search_relevant_chunks", search)

    def second(model, messages):
        assert isinstance(messages[-1], ToolMessage)
        assert json.loads(messages[-1].content)["output"]["evidence"][0]["metadata"]["chunk_id"] == 1
        return call(cid="b", query="B 申请材料 办理时间", top_k=7)

    result, trace, model = run(monkeypatch, [call(query="A 申请条件", top_k=3), second, finish], question=question)
    assert [c.kwargs["top_k"] for c in search.call_args_list] == [3, 7]
    assert [c.args[0] for c in search.call_args_list] == ["A 申请条件", "B 申请材料 办理时间"]
    assert trace.retrieval_call_count == trace.tool_call_count == 2
    assert trace.llm_call_count == 3
    assert trace.retrieved_chunk_ids == (1, 2, 3)
    assert trace.reference_chunk_ids == (1, 3)
    assert [ref["chunk_id"] for ref in result["references"]] == [1, 3]
    assert [m.tool_call_id for m in model.received_messages[2] if isinstance(m, ToolMessage)] == ["a", "b"]
    assert all(c.tool_mode is ToolMode.RAG_ONLY and c.permissions == ("knowledge_base:read",) and not c.session_id for c in setup_prediction)
    assert "web_search" not in model.bound_tool_names


@pytest.mark.parametrize("tool,args,expected", [
    ("search_knowledge_base", {"query": "申请条件", "top_k": 3}, (1, 2)),
    ("compare_documents", {"query": "申请条件", "doc_a": "A.md", "doc_b": "B.md", "top_k": 3}, (1, 2, 3)),
    ("get_chunk_text", {"chunk_id": 1}, (1,)),
])
def test_all_kb_evidence_tools_contribute(monkeypatch, setup_prediction, tool, args, expected):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [chunk(1), chunk(2)])
    _, trace, _ = run(monkeypatch, [call(tool, **args), finish])
    assert trace.retrieved_chunk_ids == expected
    assert trace.retrieval_call_count == 1


def test_parallel_completion_order_does_not_reorder_observations(monkeypatch, setup_prediction):
    def search(query, top_k):
        if query == "A":
            time.sleep(0.04)
        return [chunk(1 if query == "A" else 3)]

    monkeypatch.setattr(pipeline, "search_relevant_chunks", search)
    calls = call(query="A", top_k=3)
    calls.tool_calls.extend(call(cid="b", query="B", top_k=7).tool_calls)
    _, trace, _ = run(monkeypatch, [calls, finish])
    assert trace.retrieved_chunk_ids == (1, 3)


def test_no_relevant_info_is_valid_prediction(monkeypatch, setup_prediction):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [])
    _, trace, _ = run(monkeypatch, [call(query="缺失资料", top_k=3), FinalAgentResponse(outcome="no_evidence", answer="文档没有相关信息。")])
    assert trace.status == "NO_RELEVANT_INFO"
    assert trace.retrieved_chunk_ids == trace.reference_chunk_ids == ()
    assert trace.retrieval_call_count == 1


def test_partial_answer_keeps_only_supported_reference(monkeypatch, setup_prediction):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [chunk(1)])
    _, trace, _ = run(monkeypatch, [call(query="A B 条件", top_k=3), finish])
    assert trace.status == "OK" and trace.reference_chunk_ids == (1,)


def test_rag_only_denies_forged_web_call(monkeypatch, setup_prediction):
    with pytest.raises(RuntimeError, match="TOOL_ERROR"):
        run(monkeypatch, [call("web_search", query="test", max_results=1, freshness="any")])


@pytest.mark.parametrize("cid", [None, "not-an-id", 999])
def test_invalid_or_unmapped_kb_id_fails_closed(monkeypatch, setup_prediction, cid):
    evidence = {**chunk(cid), "source_id": "stable", "position": 0}
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [evidence])
    with pytest.raises(ValueError, match="chunk_id"):
        run(monkeypatch, [call(query="条件", top_k=3), FinalAgentResponse(outcome="no_evidence", answer="信息不足。")])


@pytest.mark.parametrize("status", ["MODEL_ERROR", "TOOL_ERROR", "INTERNAL_ERROR"])
def test_error_status_never_writes_prediction(monkeypatch, tmp_path, status):
    from app.services.chat_models import ChatServiceOutcome
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", lambda *a, **k: ChatServiceOutcome(
        http_status=500, code=500, message="error", question="问题", result={"status": status, "answer": "unsafe"}))
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([{"qid": "q", "question": "问题"}]))
    output = tmp_path / "prediction.csv"
    with pytest.raises(RuntimeError, match=status):
        rag_predictions.run_predictions(eval_file=eval_file, out_file=output)
    assert not output.exists()


def test_v3_csv_uses_safe_metadata(monkeypatch, tmp_path, setup_prediction):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [chunk(1), chunk(2)])
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    model = FakeToolCallingModel([call(query="条件", top_k=3), finish])
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([{"qid": "q", "question": "申请条件是什么"}]))
    output = tmp_path / "prediction.csv"
    rag_predictions.run_predictions(eval_file=eval_file, out_file=output)
    row = next(csv.DictReader(output.open(encoding="utf-8-sig")))
    assert row["prediction_schema_version"] == "3"
    assert row["retrieved_chunk_ids"] == "1,2" and row["reference_chunk_ids"] == "1"
    assert set(row) == set(rag_predictions.PREDICTION_FIELDNAMES)
    assert not {"context_chunk_ids", "final_chunk_ids", "top_k"} & row.keys()
    assert "证据正文" not in output.read_text(encoding="utf-8-sig")


def test_capture_excludes_external_and_failed_evidence(monkeypatch):
    from types import SimpleNamespace
    payload = {"ok": True, "output": {"evidence": [
        {"page_content": "private", "metadata": {"source_type": "web", "chunk_id": 8}},
        {"page_content": "private", "metadata": {"source_type": "knowledge_base", "chunk_id": 1}},
    ]}}
    monkeypatch.setattr(framework_tools, "execute_project_tool", lambda **k: payload)
    with capture_tool_output_keys() as records:
        framework_tools.execute_project_tool(spec=SimpleNamespace(name="example"), tool_args={})
        payload["ok"] = False
        framework_tools.execute_project_tool(spec=SimpleNamespace(name="example"), tool_args={})
    assert [r["kb_chunk_ids"] for r in records] == [[1], []]
    assert "private" not in str(records)


def test_runner_cutoff_only_reaches_scorer(monkeypatch, tmp_path):
    captured = {}
    def predict(**kwargs):
        captured["predict"] = kwargs
        return [{"prediction_schema_version": 3}]
    def evaluate(**kwargs):
        captured["evaluate"] = kwargs
        return {}
    monkeypatch.setattr(rag_predictions, "run_predictions", predict)
    monkeypatch.setattr("scripts.evaluation.rag_evaluator.run_rag_evaluation", evaluate)
    runner.run_rag_baseline(rag_eval_path=tmp_path / "q.json", output_dir=tmp_path,
                            retrieval_k=10, force_rebuild=False, use_llm_judge=False, judge_model=None)
    assert captured["evaluate"]["k"] == 10
    assert set(captured["predict"]) == {"eval_file", "out_file", "force_rebuild"}


@pytest.mark.parametrize("option", ["--top-k", "--decompose-top-k"])
def test_old_cli_parameters_are_rejected(option):
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([option, "5"])


def test_need_clarification_is_valid_prediction(monkeypatch, setup_prediction):
    _, trace, _ = run(monkeypatch, [FinalAgentResponse(
        outcome="clarify", answer="请补充申请对象。", clarification_question="申请对象是 A 还是 B？",
    )])
    assert trace.status == "NEED_CLARIFICATION"
    assert trace.retrieval_call_count == 0
    assert trace.retrieved_chunk_ids == trace.reference_chunk_ids == ()
