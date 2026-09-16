"""统一 Evaluation Runner 与多轮内存 session 测试。"""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.agent import framework_tools
from app.agent.results import create_agent_result
from app.config import AppSettings
from app.contracts.chat import AgentResultStatus
from app.agent.streaming import AgentRuntimeStreamEvent
from app.services.chat_service import ChatServiceDependencies
from app.tools.context import PrincipalType, ToolInvocationContext
from tests.support.local_tools import visible_tools
from app.tools.catalog import get_local_tool_specs
from scripts.evaluation import agent as agent_evaluation
from scripts.evaluation import runner
from scripts.evaluation.contracts import AgentEvaluationDataset


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "evaluation"
    / "minimal_agent_cases_v2.json"
)
EXTENDED_DATASET = (
    Path(__file__).parents[1]
    / "data"
    / "eval"
    / "agent_eval_cases_extended_v2.json"
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
            original_question="请列出知识库文档",
            effective_question="请列出知识库文档",
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
            tool.name for tool in visible_tools(resolve_agent_tools(context), context=context)
        ]

    assert visible_names == [
        "search_knowledge_base",
        "compare_documents",
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


def test_extended_validate_only_never_initializes_mcp(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(runner, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        runner,
        "evaluation_runtime_scope",
        lambda *args, **kwargs: pytest.fail("MCP runtime 不应初始化"),
    )

    code = runner.main(
        [
            "--validate-only",
            "--include-mcp",
            "--include-mixed",
            "--agent-cases",
            str(EXTENDED_DATASET),
        ]
    )

    assert code == 0


def test_suite_permissions_are_case_specific_and_mixed_adds_web():
    permissions = agent_evaluation._permissions_for_suite

    assert permissions("core") == ("knowledge_base:read",)
    assert permissions("online") == (
        "knowledge_base:read",
        "web:search",
        "web:fetch",
    )
    assert permissions("mcp", ("github",)) == (
        "knowledge_base:read",
        "mcp:github",
    )
    assert permissions("mixed", ("github", "firecrawl")) == (
        "knowledge_base:read",
        "web:search",
        "web:fetch",
        "mcp:github",
        "mcp:firecrawl",
    )


def test_online_runtime_scope_does_not_initialize_mcp(
    tmp_path,
    monkeypatch,
):
    from app.tools.mcp import client as mcp_client

    settings = runner.build_evaluation_settings(
        _settings(tmp_path).model_copy(update={"mcp_tools_enabled": True}),
        tmp_path / "online",
        mcp_enabled=False,
    )

    async def forbidden_initialize(_settings):
        pytest.fail("Online suite 不应初始化 MCP")

    monkeypatch.setattr(mcp_client, "initialize_mcp_runtime", forbidden_initialize)

    with runner.evaluation_runtime_scope(settings) as catalog:
        assert catalog == []


def test_mcp_runtime_initialize_and_close_are_paired_after_failure(
    tmp_path,
    monkeypatch,
):
    from app.tools.mcp import client as mcp_client
    from app.tools.mcp import provider as mcp_provider

    events = []
    settings = runner.build_evaluation_settings(
        _settings(tmp_path),
        tmp_path / "mcp",
        mcp_enabled=True,
    )

    @contextmanager
    def no_scope(_settings):
        yield

    async def fake_initialize(_settings):
        events.append("initialize")

    async def fake_close():
        events.append("close")

    monkeypatch.setattr(runner, "evaluation_settings_scope", no_scope)
    monkeypatch.setattr(mcp_client, "initialize_mcp_runtime", fake_initialize)
    monkeypatch.setattr(mcp_client, "close_mcp_runtime", fake_close)
    monkeypatch.setattr(
        mcp_provider,
        "get_mcp_tool_catalog",
        lambda: [
            {
                "server": "github",
                "tool_name": "github_get_file_contents",
                "remote_tool": "get_file_contents",
                "risk_level": "read_only",
                "approval_required": False,
                "enabled": True,
                "enabled_for_default_principal": False,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="case failed"):
        with runner.evaluation_runtime_scope(settings) as catalog:
            assert catalog[0]["server"] == "github"
            raise RuntimeError("case failed")

    assert events == ["initialize", "close"]


@pytest.mark.parametrize(
    "profiles",
    [
        (("core", False), ("mcp", True)),
        (("mcp", True), ("core", False)),
        (("mcp", True), ("online", False)),
        (("mixed", True), ("core", False)),
    ],
    ids=["core-to-mcp", "mcp-to-core", "mcp-to-online", "mixed-to-core"],
)
def test_evaluation_profiles_do_not_share_mcp_provider_cache(
    profiles,
    tmp_path,
    monkeypatch,
):
    from app.tools.mcp import client as mcp_client
    from app.tools.mcp.provider import (
        McpToolProvider,
        get_mcp_tool_provider,
        install_mcp_tool_provider,
    )

    async def fake_initialize(settings):
        install_mcp_tool_provider(McpToolProvider([], settings=settings))

    async def fake_close():
        mcp_client.reset_mcp_runtime()

    monkeypatch.setattr(mcp_client, "initialize_mcp_runtime", fake_initialize)
    monkeypatch.setattr(mcp_client, "close_mcp_runtime", fake_close)

    for suite, enabled in profiles:
        settings = runner.build_evaluation_settings(
            _settings(tmp_path),
            tmp_path / f"profile-{suite}",
            mcp_enabled=enabled,
        )
        with runner.evaluation_runtime_scope(settings):
            assert (get_mcp_tool_provider() is not None) is enabled
        assert get_mcp_tool_provider() is None


def test_capture_tool_output_keys_never_keeps_values(monkeypatch):
    def fake_execute(*, spec, tool_args, **kwargs):
        del tool_args, kwargs
        return {
            "status": "success",
            "ok": True,
            "tool_name": spec.name,
            "output": {"count": 7, "private_value": "must-not-persist"},
        }

    monkeypatch.setattr(
        framework_tools,
        "execute_project_tool",
        fake_execute,
    )

    with agent_evaluation.capture_tool_output_keys() as records:
        spec = next((item for item in get_local_tool_specs(get_settings()) if item.name == "list_documents"), None)
        assert spec is not None
        framework_tools.execute_project_tool(
            spec=spec,
            tool_args={},
        )

    assert records == [
        {
            "tool_name": "list_documents",
            "status": "success",
            "ok": True,
            "output_keys": ["count", "private_value"],
        }
    ]
    assert "must-not-persist" not in repr(records)


def test_in_memory_multi_turn_uses_chat_service_history_and_isolation(
    tmp_path,
    monkeypatch,
):
    from app.agent import streaming as agent_streaming
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

    async def fake_runtime(**kwargs):
        yield AgentRuntimeStreamEvent("result", {"result": create_agent_result(
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
        )})

    dependencies = ChatServiceDependencies(
        settings_provider=lambda: settings,
        get_history=lambda session_id: [],
        rewrite_query_with_history=fake_rewrite,
        stream_agent=fake_runtime,
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
        agent_streaming,
        "stream_agent",
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
        "app.rag.pipeline.get_rag_status",
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
