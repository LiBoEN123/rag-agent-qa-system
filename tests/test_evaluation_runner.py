"""统一 Evaluation Runner 与多轮内存 session 测试。"""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from app.agent.results import create_agent_result
from app.config import AppSettings
from app.contracts.chat import AgentResultStatus
from app.services.chat_service import ChatServiceDependencies
from app.tools import runtime as tool_runtime
from app.tools.context import PrincipalType, ToolInvocationContext
from app.tools.registry import ToolRegistry
from scripts.evaluation import agent as agent_evaluation
from scripts.evaluation import runner
from scripts.evaluation.contracts import AgentEvaluationDataset


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation"
    / "minimal_agent_cases_v2.json"
)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        _env_file=None,
        sqlite_db_path=tmp_path / "production.sqlite",
        metrics_path=tmp_path / "production-metrics.jsonl",
        bad_cases_path=tmp_path / "production-bad.jsonl",
        tool_audit_path=tmp_path / "production-audit.jsonl",
    )


def test_core_evaluation_snapshot_explicitly_disables_production_mcp(tmp_path):
    production = _settings(tmp_path).model_copy(
        update={"mcp_tools_enabled": True}
    )

    evaluation = runner.build_evaluation_settings(
        production,
        tmp_path / "evaluation",
    )

    assert production.mcp_tools_enabled is True
    assert evaluation.mcp_tools_enabled is False


def test_core_agent_builds_when_environment_enables_mcp(
    tmp_path,
    settings_env,
):
    from app.agent.runtime import prepare_agent_runtime_invocation

    production = settings_env(mcp_tools_enabled=True)
    evaluation = runner.build_evaluation_settings(
        production,
        tmp_path / "evaluation",
    )

    with runner.evaluation_settings_scope(evaluation):
        invocation = prepare_agent_runtime_invocation(
            original_question="请计算 1+1",
            effective_question="请计算 1+1",
            top_k=5,
            decompose_top_k=10,
            trace_id="evaluation-mcp-disabled",
            session_id="evaluation-mcp-disabled",
            tool_context=ToolInvocationContext(
                trace_id="evaluation-mcp-disabled",
                session_id="evaluation-mcp-disabled",
                principal_id="evaluation_baseline_v2",
                principal_type=PrincipalType.SERVICE,
                permissions=("knowledge_base:read",),
                source="evaluation_baseline_v2",
            ),
            settings=evaluation,
        )

    assert invocation.agent is not None


def test_core_evaluation_exposes_only_authorized_local_tools(
    tmp_path,
    settings_env,
):
    from app.agent.framework_tools import resolve_agent_tools

    production = settings_env(mcp_tools_enabled=True)
    evaluation = runner.build_evaluation_settings(
        production,
        tmp_path / "evaluation",
    )
    context = ToolInvocationContext(
        principal_id="evaluation_baseline_v2",
        principal_type=PrincipalType.SERVICE,
        permissions=("knowledge_base:read",),
        source="evaluation_baseline_v2",
    )

    with runner.evaluation_settings_scope(evaluation):
        visible_names = [
            tool.name for tool in resolve_agent_tools(context)
        ]

    assert visible_names == [
        "search_knowledge_base",
        "compare_documents",
        "simple_calculator",
        "list_documents",
        "get_chunk_text",
    ]


def test_validate_only_never_runs_agent_rag_or_creates_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(runner, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        runner,
        "run_agent_baseline",
        lambda *args, **kwargs: pytest.fail("Agent 不应运行"),
    )
    monkeypatch.setattr(
        runner,
        "run_rag_baseline",
        lambda *args, **kwargs: pytest.fail("RAG 不应运行"),
    )
    output = tmp_path / "output"

    code = runner.main(
        [
            "--validate-only",
            "--agent-cases",
            str(FIXTURE),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert not output.exists()
    assert "validation passed" in capsys.readouterr().out


def test_capture_tool_output_keys_never_keeps_values(monkeypatch):
    def fake_execute(*, registry, tool_name, tool_args=None, context=None):
        del registry, tool_args, context
        return {
            "status": "success",
            "ok": True,
            "tool_name": tool_name,
            "output": {"result": 7, "private_value": "must-not-persist"},
        }

    monkeypatch.setattr(
        tool_runtime,
        "execute_registered_tool",
        fake_execute,
    )

    with agent_evaluation.capture_tool_output_keys() as records:
        tool_runtime.execute_registered_tool(
            registry=ToolRegistry(),
            tool_name="simple_calculator",
            tool_args={"expression": "1+2"},
        )

    assert records == [
        {
            "tool_name": "simple_calculator",
            "status": "success",
            "ok": True,
            "output_keys": ["private_value", "result"],
        }
    ]
    assert "must-not-persist" not in repr(records)


def test_in_memory_multi_turn_uses_chat_service_history_and_isolation(
    tmp_path,
    monkeypatch,
):
    from app.agent import runtime as agent_runtime
    from app.services import chat_service

    settings = _settings(tmp_path)
    histories: list[list[dict]] = []

    def fake_rewrite(question, history, trace_id=""):
        del trace_id
        histories.append([dict(item) for item in history])
        rewritten = bool(history)
        return {
            "original_question": question,
            "rewritten_question": (
                f"第一轮背景 {question}" if rewritten else question
            ),
            "rewritten": rewritten,
            "rewrite_llm_used": rewritten,
            "history_used": rewritten,
            "reason": "fake",
        }

    def fake_runtime(**kwargs):
        return create_agent_result(
            status=AgentResultStatus.OK,
            answer="回答",
            action="agent",
            tool_trace=[
                {
                    "step": "decide",
                    "decision_type": "use_agent",
                    "action": "agent",
                }
            ],
            answer_llm_used=True,
            llm_call_count=1,
        )

    dependencies = ChatServiceDependencies(
        settings_provider=lambda: settings,
        get_history=lambda session_id: [],
        rewrite_query_with_history=fake_rewrite,
        run_agent=fake_runtime,
        log_and_write_metrics=lambda item: None,
        write_bad_cases=lambda item: None,
        append_message=lambda session_id, role, content: None,
        now_iso=lambda: "2026-01-01T00:00:00+00:00",
        perf_counter=__import__("time").perf_counter,
    )
    monkeypatch.setattr(
        chat_service,
        "get_default_chat_service_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        agent_runtime,
        "run_agent",
        fake_runtime,
    )
    dataset = AgentEvaluationDataset.model_validate(
        {
            "dataset_version": 2,
            "name": "multi",
            "cases": [
                {
                    "id": "multi",
                    "version": 2,
                    "suite": "core",
                    "category": "multi_turn",
                    "turns": [
                        {
                            "question": "第一轮",
                            "expected": {
                                "decision": "use_agent",
                                "action": "agent",
                                "status": "OK",
                            },
                        },
                        {
                            "question": "第二轮",
                            "expected": {
                                "decision": "use_agent",
                                "action": "agent",
                                "status": "OK",
                                "rewrite_keywords": ["第一轮"],
                                "min_rewrite_keyword_hits": 1,
                                "rewritten": True,
                            },
                        },
                    ],
                }
            ],
        }
    )

    first = agent_evaluation.execute_agent_case(
        dataset.cases[0],
        settings=settings,
        top_k=5,
        decompose_top_k=10,
    )
    second = agent_evaluation.execute_agent_case(
        dataset.cases[0],
        settings=settings,
        top_k=5,
        decompose_top_k=10,
    )

    assert first["passed"] is True
    assert second["passed"] is True
    assert histories[0] == []
    assert [item["role"] for item in histories[1]] == [
        "user",
        "assistant",
    ]
    assert histories[2] == []
    assert not settings.sqlite_db_path.exists()
    assert not settings.metrics_path.exists()
    assert not settings.bad_cases_path.exists()
    assert not settings.tool_audit_path.exists()


def _fake_manifest(run_id: str):
    return {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "code": {
            "git_commit": "abc",
            "git_branch": "branch",
            "git_dirty": False,
            "python_version": "3.12",
            "platform": "test",
        },
        "data": {
            "agent_cases_path": "fixture",
            "agent_cases_sha256": "a",
            "rag_eval_path": "fixture",
            "rag_eval_sha256": "b",
        },
        "rag": {
            "index_source": "fake",
            "document_count": 1,
            "chunk_count": 1,
            "embedding_model": "fake",
            "rerank_model": "fake",
        },
        "agent": {
            "answer_model": "fake",
        },
        "judge": {"enabled": False, "model": "", "cache_source": ""},
    }


def test_runner_writes_expected_output_structure(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    output = tmp_path / "run"
    rag_call = {}

    @contextmanager
    def no_scope(_settings):
        yield

    agent_summary = {
        "pass_rate": 1.0,
        "first_tool_accuracy": 1.0,
        "status_accuracy": 1.0,
        "source_pass_rate": 1.0,
        "multi_turn_pass_rate": 0.0,
        "p50_latency_ms": 1,
        "p95_latency_ms": 1,
        "average_llm_call_count": 1,
        "average_tool_call_count": 1,
        "groups": {
            "suite": {
                "core": {
                    "total_cases": 1,
                    "passed_cases": 1,
                    "pass_rate": 1.0,
                }
            }
        },
    }
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runner,
        "get_git_state",
        lambda: {
            "git_commit": "abc",
            "git_branch": "branch",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(runner, "evaluation_settings_scope", no_scope)
    monkeypatch.setattr(
        runner,
        "run_agent_baseline",
        lambda *args, **kwargs: ([], agent_summary),
    )
    def fake_rag_baseline(**kwargs):
        rag_call.update(kwargs)
        return {
            "rows": [],
            "summary": {"overall": {"prediction_coverage": 1.0}},
            "judge_enabled": False,
            "prediction_metadata": {
                "prediction_schema_version": 2,
                "effective_top_k": kwargs["top_k"],
                "effective_decompose_top_k": kwargs[
                    "decompose_top_k"
                ],
            },
        }

    monkeypatch.setattr(
        runner,
        "run_rag_baseline",
        fake_rag_baseline,
    )
    monkeypatch.setattr(
        runner,
        "build_manifest",
        lambda **kwargs: _fake_manifest(kwargs["run_id"]),
    )
    monkeypatch.setattr(
        "app.rag.get_rag_status",
        lambda: {},
    )

    code = runner.main(
        [
            "--agent-cases",
            str(FIXTURE),
            "--rag-eval-file",
            str(FIXTURE),
            "--output-dir",
            str(output),
            "--top-k",
            "3",
            "--decompose-top-k",
            "7",
        ]
    )

    assert code == 0
    assert {
        "manifest.json",
        "baseline_summary.json",
        "baseline_report.md",
        "coverage_report.json",
        "failure_cases.jsonl",
        "agent/per_case.jsonl",
        "agent/per_case.csv",
        "agent/summary.json",
    } <= {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    report = (output / "baseline_report.md").read_text(encoding="utf-8")
    assert "LLM Judge skipped" in report
    assert rag_call["top_k"] == 3
    assert rag_call["decompose_top_k"] == 7


def test_dirty_worktree_requires_explicit_allow_dirty(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(runner, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        runner,
        "get_git_state",
        lambda: {
            "git_commit": "abc",
            "git_branch": "branch",
            "git_dirty": True,
        },
    )

    with pytest.raises(SystemExit) as exc:
        runner.main(
            [
                "--agent-cases",
                str(FIXTURE),
                "--agent-only",
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exc.value.code == 2


def test_unified_rag_predictions_fail_fast_on_model_error(
    tmp_path,
    monkeypatch,
):
    from scripts.evaluation import rag_predictions

    eval_file = tmp_path / "rag.json"
    eval_file.write_text(
        json.dumps([{"qid": "q1", "question": "问题"}]),
        encoding="utf-8",
    )
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        rag_predictions,
        "initialize_rag_for_predictions",
        lambda force_rebuild=False: None,
    )
    monkeypatch.setattr(
        rag_predictions,
        "run_rag_once_with_result",
        lambda question, top_k, decompose_top_k: (
            {
                "status": "MODEL_ERROR",
                "answer": "",
            },
            rag_predictions.RagPredictionTrace(
                final_chunk_ids=(),
                context_chunk_ids=(),
                reference_chunk_ids=(),
                context_char_count=0,
                top_k=top_k,
                decompose_top_k=decompose_top_k,
                decompose_llm_used=False,
                status="MODEL_ERROR",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="qid=q1"):
        rag_predictions.run_predictions(
            eval_file=eval_file,
            out_file=output,
            fail_on_model_error=True,
        )

    assert not output.exists()
