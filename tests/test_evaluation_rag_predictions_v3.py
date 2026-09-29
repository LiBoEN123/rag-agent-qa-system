"""Prediction v3 通过真实 ChatService / create_agent / governed tools 的确定性验收。"""

import asyncio
import csv
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import framework_agent, framework_tools
from app.agent.final_response import FinalAgentResponse
from app.contracts.chat import ToolMode
from app.rag import runtime as pipeline
from scripts.evaluation import rag_predictions, runner
from scripts.evaluation.agent import capture_tool_output_keys
from tests.support.agent_runtime import FakeToolCallingModel
from tests.support.answer_streaming import ControlledAnswerModel


def chunk(cid, file="A.md"):
    return {"chunk_id": cid, "file_name": file, "text": f"证据正文 {cid}", "hybrid_score": 0.9, "metadata": {"chunk_id": cid, "file_name": file, "source_id": file, "position": cid}}


def call(name="search_knowledge_base", cid="a", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def finish(model, messages):
    docs = [doc for message in messages if isinstance(message, ToolMessage)
            for doc in (message.artifact or json.loads(message.content)).get("output", {}).get("evidence", [])]
    return model.final_response(scope_match="matched", support_level="full", outcome="answer",
                                evidence_ids=[doc["id"] for doc in docs if doc["metadata"]["chunk_id"] in (1, 3)])


@pytest.fixture
def setup_prediction(monkeypatch):
    composer = ControlledAnswerModel(first_text="A 与 B 的申请条件", last_text="见已检索证据。")
    composer.release.set()
    monkeypatch.setattr("app.agent.streaming.get_answer_model", lambda: composer)
    monkeypatch.setattr(pipeline, "RAG_READY", True)
    monkeypatch.setattr(pipeline, "chunks", [chunk(1), chunk(2), chunk(3, "B.md")])
    monkeypatch.setattr(pipeline, "rerank", lambda chunks, query: chunks)
    contexts = []
    original = framework_tools.execute_project_tool

    async def execute(**kwargs):
        contexts.append(kwargs["context"])
        return await original(**kwargs)

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
    assert trace.llm_call_count == 4
    assert trace.retrieved_chunk_ids == (1, 2, 3)
    assert trace.reference_chunk_ids == (1, 3)
    assert [ref["chunk_id"] for ref in result["references"]] == [1, 3]
    assert [m.tool_call_id for m in model.received_messages[2] if isinstance(m, ToolMessage)] == ["a", "b"]
    assert all(c.tool_mode is ToolMode.RAG_ONLY and c.permissions == ("knowledge_base:read",) and not c.session_id for c in setup_prediction)
    assert "web_search" not in model.bound_tool_names


@pytest.mark.parametrize("tool,args,expected", [
    ("search_knowledge_base", {"query": "申请条件", "top_k": 3}, (1, 2)),
    ("search_document", {"query": "申请条件", "document": "A.md", "top_k": 3}, (1, 2)),
    ("get_chunk_context", {"chunk_id": 1, "before": 0, "after": 0}, (1,)),
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
    _, trace, _ = run(monkeypatch, [call(query="缺失资料", top_k=3), FinalAgentResponse(scope_match="matched", support_level="none", outcome="no_evidence", answer="文档没有相关信息。")])
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
        run(monkeypatch, [call(query="条件", top_k=3), FinalAgentResponse(scope_match="matched", support_level="none", outcome="no_evidence", answer="信息不足。")])


@pytest.mark.parametrize("status", ["MODEL_ERROR", "MODEL_BAD_OUTPUT", "TOOL_ERROR", "INTERNAL_ERROR"])
def test_error_status_never_writes_prediction(monkeypatch, tmp_path, status):
    from app.services.chat_models import ChatServiceOutcome
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", lambda *a, **k: ChatServiceOutcome(
        http_status=500, code=500, message="error", question="问题", result={"status": status, "answer": "unsafe"}))
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([{"qid": "q", "question": "问题"}]))
    output = tmp_path / "prediction.csv"
    with pytest.raises(RuntimeError, match=status):
        rag_predictions.run_rag_once_with_result("问题")
    assert rag_predictions.run_predictions(eval_file=eval_file, out_file=output) == []
    with output.open(encoding="utf-8-sig") as file:
        assert list(csv.DictReader(file)) == []
    failure_path = tmp_path / "prediction_failures.jsonl"
    failure, = [json.loads(line) for line in failure_path.read_text().splitlines()]
    assert failure == {
        "qid": "q", "question": "问题", "status": status, "reason": "",
        "retrieved_chunk_ids": [], "reference_chunk_ids": [],
        "retrieval_call_count": 0, "tool_call_count": 0, "llm_call_count": 0,
        "requested_evidence_ids": [], "available_evidence_ids": [], "unavailable_evidence_ids": [],
    }
    assert "unsafe" not in output.read_text() + failure_path.read_text()


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


def test_batch_isolates_grounding_failure_and_flushes_before_next_case(monkeypatch, tmp_path, setup_prediction):
    monkeypatch.setattr(pipeline, "search_relevant_chunks", lambda *a, **k: [chunk(1), chunk(2)])
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    models = [FakeToolCallingModel([call(query="条件", top_k=3), final]) for final in (
        finish,
        FinalAgentResponse(scope_match="matched", support_level="full", outcome="answer", answer="unsafe", evidence_ids=["knowledge_base:fake"]),
        finish,
    )]
    output = tmp_path / "predictions.csv"
    sidecar = tmp_path / "prediction_failures.jsonl"
    attempted = []
    original_chat_turn = rag_predictions.run_evaluation_chat_turn

    def chat_turn(request, **kwargs):
        # 在下一题开始时重新打开产物，证明前一题已 flush，而非仅在退出时 close。
        with output.open(encoding="utf-8-sig") as file:
            assert [row["qid"] for row in csv.DictReader(file)] == (["q1"] if attempted else [])
        assert len(sidecar.read_text().splitlines()) == (1 if len(attempted) == 2 else 0)
        attempted.append(len(attempted) + 1)
        return original_chat_turn(request, **kwargs)

    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", chat_turn)
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: models[len(attempted) - 1])
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([
        {"qid": f"q{i}", "question": f"申请条件是什么，场景 {i}"} for i in (1, 2, 3)
    ]))

    rows = rag_predictions.run_predictions(eval_file=eval_file, out_file=output)

    assert attempted == [1, 2, 3]
    assert [row["qid"] for row in rows] == ["q1", "q3"]
    with output.open(encoding="utf-8-sig") as file:
        assert [row["qid"] for row in csv.DictReader(file)] == ["q1", "q3"]
    failure, = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert failure["qid"] == "q2" and failure["status"] == "MODEL_ERROR"
    assert failure["reason"] == "Agent structured response referenced unavailable evidence"
    assert failure["retrieved_chunk_ids"] == [1, 2]
    assert failure["reference_chunk_ids"] == []
    assert failure["retrieval_call_count"] == failure["tool_call_count"] == 1
    assert failure["llm_call_count"] == 2
    assert failure["requested_evidence_ids"] == failure["unavailable_evidence_ids"] == ["knowledge_base:fake"]
    assert len(failure["available_evidence_ids"]) == 2
    assert "unsafe" not in output.read_text() + sidecar.read_text()
    assert "证据正文" not in sidecar.read_text()


@pytest.mark.parametrize("damage", ["chunk_id", "alignment", "duplicate_calls"])
def test_structural_error_on_error_status_still_aborts_batch(monkeypatch, tmp_path, setup_prediction, damage):
    from types import SimpleNamespace
    from app.services.chat_models import ChatServiceOutcome

    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    monkeypatch.setattr(framework_tools, "execute_project_tool", AsyncMock(return_value={
        "tool_call_id": "a", "ok": True, "output": {"evidence": [
            {"metadata": {"source_type": "knowledge_base", "chunk_id": None if damage == "chunk_id" else 1}},
        ]},
    }))
    attempted = []

    def chat_turn(request, **kwargs):
        attempted.append(request.question)
        if len(attempted) == 1:
            result = {"status": "OK", "answer": "已完成"}
        else:
            for _ in range(2 if damage == "duplicate_calls" else 1):
                asyncio.run(framework_tools.execute_project_tool(spec=SimpleNamespace(name="search_knowledge_base"), tool_args={}))
            result = {"status": "MODEL_ERROR", "answer": "unsafe", "tool_trace": [
                {"step": "tool_call", "tool_call_id": "missing" if damage == "alignment" else "a"},
            ]}
        return ChatServiceOutcome(http_status=200, code=200, message="test", question=request.question, result=result)

    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", chat_turn)
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([{"qid": f"q{i}", "question": f"问题{i}"} for i in (1, 2, 3)]))
    output = tmp_path / "predictions.csv"
    with pytest.raises(RuntimeError, match="qid=q2"):
        rag_predictions.run_predictions(eval_file=eval_file, out_file=output)
    assert attempted == ["问题1", "问题2"]
    with output.open(encoding="utf-8-sig") as file:
        assert [row["qid"] for row in csv.DictReader(file)] == ["q1"]
    assert (tmp_path / "prediction_failures.jsonl").read_text() == ""


@pytest.mark.parametrize("data", [[], {"items": {}}, [{"qid": "q"}], [
    {"qid": "q", "question": "问题"}, {"qid": "q", "question": "重复"},
]])
def test_corrupt_dataset_fails_before_prediction(monkeypatch, tmp_path, data):
    initialize = Mock()
    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", initialize)
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        rag_predictions.run_predictions(eval_file=eval_file, out_file=tmp_path / "predictions.csv")
    initialize.assert_not_called()


def test_capture_excludes_external_and_failed_evidence(monkeypatch):
    from types import SimpleNamespace
    payload = {"ok": True, "output": {"evidence": [
        {"page_content": "private", "metadata": {"source_type": "web", "chunk_id": 8}},
        {"page_content": "private", "metadata": {"source_type": "knowledge_base", "chunk_id": 1}},
    ]}}
    monkeypatch.setattr(framework_tools, "execute_project_tool", AsyncMock(return_value=payload))
    with capture_tool_output_keys() as records:
        asyncio.run(framework_tools.execute_project_tool(spec=SimpleNamespace(name="example"), tool_args={}))
        payload["ok"] = False
        asyncio.run(framework_tools.execute_project_tool(spec=SimpleNamespace(name="example"), tool_args={}))
    assert [r["kb_chunk_ids"] for r in records] == [[1], []]
    assert "private" not in str(records)


def test_runner_cutoff_only_reaches_scorer(monkeypatch, tmp_path):
    captured = {}
    def predict(**kwargs):
        captured["predict"] = kwargs
        path = kwargs["out_file"].with_name("prediction_failures.jsonl")
        path.parent.mkdir(parents=True)
        path.write_text("")
        return [{"prediction_schema_version": 3}]
    def evaluate(**kwargs):
        captured["evaluate"] = kwargs
        return {"summary": {}}
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
        scope_match="matched", support_level="none",
        outcome="clarify", answer="请补充申请对象。", clarification_question="申请对象是 A 还是 B？",
    )])
    assert trace.status == "NEED_CLARIFICATION"
    assert trace.retrieval_call_count == 0
    assert trace.retrieved_chunk_ids == trace.reference_chunk_ids == ()
