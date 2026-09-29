"""GEval、空值和只读缓存合同；使用原生 metric 与 fake 项目模型，禁止网络。"""

import asyncio
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from scripts.evaluation import deepeval_adapter as adapter, rag_evaluator
from scripts.evaluation.rag.common import ChunkIndex
from scripts.evaluation.rag.evaluation import evaluate
from scripts.evaluation.rag.report import write_csv


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("unexpected network, cloud or real model")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr("deepeval.metrics.api.metric_data_manager.post_metric_if_enabled", deny)
    monkeypatch.setattr("app.llm.factory._create_model", deny)


def question(qid="q1", behavior="answer"):
    return {"qid": qid, "question": "政策是什么？", "category": "test", "difficulty": "easy",
            "expected_behavior": behavior, "gold_answer": "已知政策，未知细节。",
            "gold_key_points": [{"point": "已知政策", "aliases": ["同义政策"]}],
            "forbidden_points": ["虚构细节"], "gold_chunk_ids": [1]}


def prediction(answer="已知政策"):
    return {"prediction_schema_version": 3, "answer": answer,
            "retrieved_chunk_ids": [1], "reference_chunk_ids": [1]}


@pytest.mark.parametrize("behavior", ["answer", "partial_answer", "abstain"])
def test_native_geval_uses_project_gateway_and_fixed_gold_spec(monkeypatch, settings_env, behavior):
    settings_env(judge_model="configured-judge", judge_timeout=7, judge_max_retries=3, judge_max_tokens=456)
    calls, options = [], {}
    def create(**kwargs):
        options.update(kwargs)
        def invoke(messages):
            calls.append(messages)
            return AIMessage(content='{"score": 8, "reason": "事实正确，有少量遗漏"}')
        return SimpleNamespace(model_name=kwargs["model_name"], invoke=invoke)
    monkeypatch.setattr("app.llm.factory._create_model", create)
    judge = adapter.ProjectJudge(model_name="cli-judge", api_url="https://judge.test/v1", api_key="fake-key")
    score, reason = adapter.answer_correctness(question(behavior=behavior), "回答", model=judge)
    assert score == 0.8 and reason == "事实正确，有少量遗漏"
    assert len(calls) == 1  # 固定 steps，无步骤生成或项目 JSON retry。
    assert isinstance(calls[0][0], HumanMessage)
    prompt = calls[0][0].content
    assert all(text in prompt for text in ("政策是什么？", "回答", behavior, "已知政策，未知细节。", "已知政策", "虚构细节"))
    assert options == dict(role="judge", model_name="cli-judge", api_url="https://judge.test/v1",
                           api_key="fake-key", temperature=0.0, timeout=7, max_retries=3, max_tokens=456)
    assert judge.get_model_name() == "cli-judge"
    assert json.loads(asyncio.run(judge.a_generate("async prompt")))["score"] == 8


@pytest.mark.parametrize("judge_enabled", [False, True])
@pytest.mark.parametrize("with_gold", [False, True])
def test_missing_prediction_is_unscored_and_never_judged(judge_enabled, with_gold):
    q = question()
    q["gold_chunk_ids"] = [1] if with_gold else []
    judge = Mock(return_value=(0.8, "correct"))
    rows, summary = evaluate([q], {}, ChunkIndex({}, {}, "none"), 5, judge if judge_enabled else None)
    assert not rows[0]["has_prediction"] and not rows[0]["judge_used"]
    assert rows[0]["answer_correctness"] is None and rows[0]["answer_correctness_reason"] == ""
    assert rows[0]["retrieval_gold_chunk_recall_at_k"] == rows[0]["reference_gold_chunk_recall"] == 0
    assert summary["overall"]["avg_answer_correctness"] is None
    assert summary["overall"]["llm_judge_coverage"] == 0
    judge.assert_not_called()


def test_summary_averages_only_successful_judgments_including_zero():
    judge = Mock(side_effect=[(0.0, "incorrect"), ValueError("judge failed"), (0.8, "correct")])
    rows, summary = evaluate([question(str(i)) for i in range(4)],
                             {str(i): prediction() for i in range(3)}, ChunkIndex({}, {}, "none"), 5, judge)
    assert [r["answer_correctness"] for r in rows] == [0.0, None, 0.8, None]
    assert [r["judge_used"] for r in rows] == [True, False, True, False]
    assert rows[1]["judge_error"] == "judge failed" and rows[1]["answer_correctness_reason"] == ""
    for group in summary.values():
        assert group["avg_answer_correctness"] == 0.4 and group["llm_judge_coverage"] == 0.5
        assert group["prediction_coverage"] == 0.75


@pytest.mark.parametrize("cache_qid,cache_answer,score,error", [
    ("q1", "已知政策", "0", ""),
    ("q1", "已知政策", "0.8", ""),
    ("q1", "旧回答", "0.8", "answer changed"),
    ("q1", "已知政策 ", "0.8", "answer changed"),
    ("q2", "已知政策", "0.8", "qid missing"),
    ("q1", "已知政策", "", "no answer correctness"),
    ("q1", "已知政策", "nan", "[0, 1]"),
])
def test_cache_reuses_only_matching_answer_without_model(tmp_path, cache_qid, cache_answer, score, error):
    eval_file, pred_file, cache = [tmp_path / name for name in ("eval.json", "pred.json", "cache.csv")]
    eval_file.write_text(json.dumps([question()]))
    pred_file.write_text(json.dumps([{"qid": "q1", **prediction()}]))
    write_csv(cache, [{"qid": cache_qid, "answer": cache_answer, "answer_correctness": score,
                       "answer_correctness_reason": "cached reason"}])
    before = cache.read_bytes()
    result = rag_evaluator.run_rag_evaluation(
        eval_file=eval_file, pred_file=pred_file, out_dir=tmp_path / "out", index_source="none",
        judge_cache_report=cache, use_llm_judge=True, judge_model="never-created",
    )
    row = result["rows"][0]
    assert row["judge_used"] is (not error)
    assert row["answer_correctness"] == (None if error else float(score))
    assert row["answer_correctness_reason"] == ("" if error else "cached reason")
    assert error in row["judge_error"] and cache.read_bytes() == before
    assert set(p.name for p in (tmp_path / "out").iterdir()) == {
        "per_question_report.csv", "summary_report.csv", "summary_report.json",
    }


@pytest.mark.parametrize("enabled,model_name", [(False, "configured"), (True, ""), (True, "configured")])
def test_runner_judge_configuration_and_nullable_reports(tmp_path, monkeypatch, enabled, model_name):
    eval_file, pred_file = tmp_path / "eval.json", tmp_path / "pred.json"
    eval_file.write_text(json.dumps([question()]))
    pred_file.write_text(json.dumps([{"qid": "q1", **prediction()}]))
    monkeypatch.setattr("app.llm.factory._create_model", lambda **kwargs: SimpleNamespace(
        model_name=kwargs["model_name"], invoke=lambda messages: AIMessage(content='{"score": 9, "reason": "correct"}'),
    ))
    result = rag_evaluator.run_rag_evaluation(
        eval_file=eval_file, pred_file=pred_file, out_dir=tmp_path / "out", index_source="none",
        use_llm_judge=enabled, judge_model=model_name,
    )
    expected = 0.9 if enabled and model_name else None
    assert result["rows"][0]["answer_correctness"] == expected
    summary = json.loads((tmp_path / "out/summary_report.json").read_text())["overall"]
    assert summary["avg_answer_correctness"] == expected
    assert summary["llm_judge_coverage"] == int(expected is not None)
