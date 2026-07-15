import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_rag_results_semantic_fixed as evaluator


def judge_config() -> evaluator.LLMJudgeConfig:
    return evaluator.LLMJudgeConfig(
        enabled=True,
        model="fake-judge",
        api_url="http://example.test/v1/chat/completions",
        api_key="",
        timeout=1,
        max_retries=0,
        retry_sleep=0,
        max_tokens=1024,
    )


def patch_judge(monkeypatch, key_point_result=None, forbidden_result=None):
    def fake_call_judge_json(system_prompt, user_prompt, config, trace_id=""):
        if system_prompt == evaluator.KEY_POINT_JUDGE_SYSTEM_PROMPT:
            result = key_point_result or {"key_point_judgments": []}
            return result, json.dumps(result, ensure_ascii=False)
        if system_prompt == evaluator.FORBIDDEN_JUDGE_SYSTEM_PROMPT:
            result = forbidden_result or {"forbidden_hit": False, "forbidden_judgments": []}
            return result, json.dumps(result, ensure_ascii=False)
        raise AssertionError("unexpected judge prompt")

    monkeypatch.setattr(evaluator, "call_judge_json", fake_call_judge_json)


def test_call_judge_llm_uses_factory_gateway_and_cli_config(monkeypatch):
    """离线 Judge 通过统一工厂和 Gateway 调用并保留 CLI 配置。"""
    captured = {}

    class FakeJudgeModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content='{"score": 1}')

    model = FakeJudgeModel()

    def fake_get_judge_model(**kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(evaluator._judge, "get_judge_model", fake_get_judge_model)
    config = evaluator.LLMJudgeConfig(
        enabled=True,
        model="cli-judge",
        api_url="https://judge.example.test/v1/chat/completions",
        api_key="cli-key",
        timeout=7,
        max_retries=3,
        retry_sleep=0.5,
        max_tokens=456,
        temperature=0.2,
    )

    result = evaluator._judge.call_judge_llm(
        "judge system",
        "judge user",
        config,
        trace_id="judge-trace",
    )

    assert result == '{"score": 1}'
    assert captured == {
        "api_key": "cli-key",
        "api_url": "https://judge.example.test/v1/chat/completions",
        "model_name": "cli-judge",
        "temperature": 0.2,
        "max_tokens": 456,
        "timeout": 7,
        "max_retries": 3,
    }
    assert isinstance(model.calls[0][0], SystemMessage)
    assert isinstance(model.calls[0][1], HumanMessage)
    assert model.calls[0][0].content == "judge system"
    assert model.calls[0][1].content == "judge user"


def test_synonym_expression_can_be_covered_by_llm_judge(monkeypatch):
    gold_points = [{"point": "其他时间不予办理。", "aliases": ["其他时间不予办理。"]}]
    answer = "学生证补办仅在每学期第 14 周办理，不是随时都能补办。"

    patch_judge(
        monkeypatch,
        key_point_result={
            "key_point_judgments": [
                {
                    "key_point": "其他时间不予办理。",
                    "status": "covered",
                    "evidence_from_answer": "不是随时都能补办",
                    "reason": "同义表达了其他时间无法办理",
                }
            ]
        },
    )

    result = evaluator.evaluate_key_points_with_llm(
        question="学生证丢失后是不是任何时候都可以补办？",
        gold_answer="其他时间不予办理。",
        gold_key_points=gold_points,
        model_answer=answer,
        judge_config=judge_config(),
    )

    assert result["key_point_recall"] == 1.0
    assert result["covered_count"] == 1
    assert result["missing_key_points"] == []


def test_negative_forbidden_mention_is_not_hit(monkeypatch):
    patch_judge(
        monkeypatch,
        forbidden_result={
            "forbidden_hit": False,
            "forbidden_judgments": [
                {
                    "forbidden_point": "具体面试题目",
                    "hit": False,
                    "answer_span": "文档没有提供具体面试题目",
                    "reason": "否定性提及，不是把面试题目当事实回答",
                }
            ],
        },
    )

    result = evaluator.evaluate_forbidden_with_llm(
        question="2025 年转专业专业复试的具体面试题目是什么？",
        model_answer="文档没有提供具体面试题目。",
        forbidden_points=["具体面试题目"],
        judge_config=judge_config(),
    )

    assert result["forbidden_hit"] is False
    assert result["forbidden_judgments"][0]["hit"] is False


def test_fabricated_forbidden_detail_is_hit(monkeypatch):
    patch_judge(
        monkeypatch,
        forbidden_result={
            "forbidden_hit": True,
            "forbidden_judgments": [
                {
                    "forbidden_point": "具体面试题目",
                    "hit": True,
                    "answer_span": "复试面试题目包括自我介绍、专业知识问答和英语口语",
                    "reason": "把未提供的具体面试题目当成事实列出",
                }
            ],
        },
    )

    result = evaluator.evaluate_forbidden_with_llm(
        question="2025 年转专业专业复试的具体面试题目是什么？",
        model_answer="复试面试题目包括自我介绍、专业知识问答和英语口语。",
        forbidden_points=["具体面试题目"],
        judge_config=judge_config(),
    )

    assert result["forbidden_hit"] is True
    assert result["forbidden_judgments"][0]["hit"] is True


def test_policy_conflict_can_be_contradicted(monkeypatch):
    gold_point = "2023 年及以后入学的普通本科四年制专业学生最长学习年限为 6 年，不能简单套用旧口径 8 年。"
    patch_judge(
        monkeypatch,
        key_point_result={
            "key_point_judgments": [
                {
                    "key_point": gold_point,
                    "status": "contradicted",
                    "evidence_from_answer": "最长学习年限为 8 年",
                    "reason": "回答直接套用旧口径 8 年，与 2023 年以后 6 年要求冲突",
                }
            ]
        },
    )

    result = evaluator.evaluate_key_points_with_llm(
        question="普通本科四年制专业学生现在最长学习年限是不是 8 年？",
        gold_answer=gold_point,
        gold_key_points=[gold_point],
        model_answer="普通本科四年制学生最长学习年限为 8 年。",
        judge_config=judge_config(),
        category="trap",
        expected_behavior="answer",
    )

    assert result["key_point_recall"] == 0.0
    assert result["contradicted_count"] == 1
    assert result["contradicted_key_points"] == [gold_point]


def test_partial_answer_should_not_get_full_credit_for_over_refusal(monkeypatch):
    gold_points = [
        "文档说明了延期毕业的申请流程。",
        "文档没有说明每次最多延期几个月。",
    ]
    patch_judge(
        monkeypatch,
        key_point_result={
            "key_point_judgments": [
                {
                    "key_point": gold_points[0],
                    "status": "missing",
                    "evidence_from_answer": "",
                    "reason": "整体说没有相关信息，遗漏了可确认的申请流程",
                },
                {
                    "key_point": gold_points[1],
                    "status": "covered",
                    "evidence_from_answer": "文档没有相关信息",
                    "reason": "表达了无法确认每次最多延期几个月",
                },
            ]
        },
    )

    result = evaluator.evaluate_key_points_with_llm(
        question="研究生延期毕业每次最多可以延期几个月？",
        gold_answer="文档说明了延期流程，但没有说明每次最多延期几个月。",
        gold_key_points=gold_points,
        model_answer="文档没有相关信息。",
        judge_config=judge_config(),
        category="partial_answer",
        expected_behavior="partial_answer",
    )

    assert result["key_point_recall"] == 0.5
    assert result["missing_key_points"] == [gold_points[0]]


def test_evaluate_uses_llm_judge_fields(monkeypatch):
    patch_judge(
        monkeypatch,
        key_point_result={
            "key_point_judgments": [
                {
                    "key_point": "其他时间不予办理。",
                    "status": "covered",
                    "evidence_from_answer": "不是随时都能补办",
                    "reason": "同义覆盖",
                }
            ]
        },
        forbidden_result={
            "forbidden_hit": False,
            "forbidden_judgments": [
                {
                    "forbidden_point": "具体面试题目",
                    "hit": False,
                    "answer_span": "文档没有提供具体面试题目",
                    "reason": "否定提及",
                }
            ],
        },
    )
    question = {
        "qid": "q-test",
        "question": "学生证丢失后是不是任何时候都可以补办？",
        "category": "trap",
        "expected_behavior": "answer",
        "gold_answer": "其他时间不予办理。",
        "gold_key_points": [{"point": "其他时间不予办理。", "aliases": ["其他时间不予办理。"]}],
        "forbidden_points": ["具体面试题目"],
    }
    preds = {"q-test": {"answer": "学生证补办仅在每学期第 14 周办理，不是随时都能补办。"}}

    rows, summary, details = evaluator.evaluate(
        [question],
        preds,
        evaluator.ChunkIndex({}, {}, "none"),
        k=5,
        judge_config=judge_config(),
    )

    row = rows[0]
    assert "old_key_point_recall" not in row
    assert "old_forbidden_hit" not in row
    assert "llm_key_point_recall" not in row
    assert "skill_type" not in row
    assert "retrieval_chunk_recall_at_k" not in row
    assert "retrieval_evidence_recall_at_k" not in row
    assert "matched_gold_chunk_ids" not in row
    assert row["key_point_recall"] == 1.0
    assert row["forbidden_hit"] is False
    assert row["key_point_judge_used"] is True
    assert row["judge_error"] == ""
    assert "avg_llm_key_point_recall" not in summary["overall"]
    assert "evidence_recall_at_k" not in summary["overall"]
    assert summary["overall"]["avg_key_point_recall"] == 1.0
    assert details[0]["key_point_recall"] == 1.0
    assert details[0]["key_point_judgments"][0]["status"] == "covered"


def test_cached_judge_evaluators_reuse_report_without_llm_calls(tmp_path):
    cache_report = tmp_path / "per_question_report.csv"
    cache_report.write_text(
        "\ufeffqid,key_point_judgments_json,forbidden_judgments_json\n"
        "\"q-cache\",\"[{\"\"key_point\"\":\"\"A\"\",\"\"status\"\":\"\"covered\"\",\"\"evidence_from_answer\"\":\"\"A\"\",\"\"reason\"\":\"\"cached\"\"}]\",\"[]\"\n",
        encoding="utf-8",
    )

    cache = evaluator.load_judge_cache_report(cache_report)
    key_point_evaluator, forbidden_evaluator = evaluator.make_cached_judge_evaluators(cache)

    key_result = key_point_evaluator(
        question="",
        gold_answer="",
        gold_key_points=["A"],
        model_answer="A",
        qid="q-cache",
    )
    forbidden_result = forbidden_evaluator(
        question="",
        model_answer="A",
        forbidden_points=[],
        qid="q-cache",
    )

    assert key_result["key_point_recall"] == 1.0
    assert key_result["key_point_judgments"][0]["reason"] == "cached"
    assert forbidden_result["forbidden_hit"] is False


def test_cached_judge_evaluator_missing_qid_fails_fast():
    key_point_evaluator, _ = evaluator.make_cached_judge_evaluators({})

    try:
        key_point_evaluator(
            question="",
            gold_answer="",
            gold_key_points=["A"],
            model_answer="A",
            qid="missing",
        )
    except evaluator.LLMJudgeError as exc:
        assert "missing from judge cache" in str(exc)
    else:
        raise AssertionError("missing qid should fail instead of calling LLM")
