"""统一 Evaluation Runner 与多轮内存 session 测试。"""

from contextlib import contextmanager
import csv
import asyncio
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


@pytest.mark.parametrize("ids", [["D008"], ["D001", "D008"], ["agent-followup-001"]])
def test_case_filter_validate_only_never_initializes_runtime(ids, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("validate-only must not initialize evaluation runtime")
    monkeypatch.setattr(runner, "evaluation_runtime_scope", forbidden)
    output = tmp_path / "not-created"
    args = ["--validate-only", "--output-dir", str(output)]
    for case_id in ids:
        args.extend(["--case-id", case_id])
    assert runner.main(args) == 0
    report = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert report["total_cases"] == len(ids)
    assert report["case_shape"]["multi_turn"] == int(ids == ["agent-followup-001"])
    assert not output.exists()


@pytest.mark.parametrize("args", [
    ["--agent-only", "--case-id", "unknown"],
    ["--validate-only", "--case-id", "unknown"],
    ["--case-id", "D008"],
    ["--rag-only", "--case-id", "D008"],
    ["--validate-only", "--case-id", "online-web-search-001"],
])
def test_case_filter_invalid_selection_fails_before_output_or_runtime(args, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid selection must fail before runtime")
    monkeypatch.setattr(runner, "evaluation_runtime_scope", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as error:
        runner.main([*args, "--output-dir", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_case_filter_executes_whole_cases_and_records_reproduction(tmp_path, monkeypatch):
    selected = []
    @contextmanager
    def no_runtime(settings):
        yield []
    def run_cases(dataset, **kwargs):
        selected.extend(dataset.cases)
        return [], {}
    def no_rag(**kwargs):
        pytest.fail("case filter must not run RAG evaluation")
    monkeypatch.setattr(runner, "evaluation_runtime_scope", no_runtime)
    monkeypatch.setattr(runner, "run_agent_baseline", run_cases)
    monkeypatch.setattr(runner, "run_rag_baseline", no_rag)
    monkeypatch.setattr(runner, "build_manifest", lambda **kwargs: _fake_manifest(kwargs["run_id"]))
    monkeypatch.setattr("app.rag.pipeline.get_rag_status", lambda: {})
    output = tmp_path / "selected"
    assert runner.main([
        "--agent-only", "--allow-dirty", "--output-dir", str(output),
        "--case-id", "D008", "--case-id", "agent-followup-001", "--case-id", "D008",
    ]) == 0
    assert [case.id for case in selected] == ["D008", "agent-followup-001"]
    assert len(selected[1].turns) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["data"]["selected_case_ids"] == ["D008", "agent-followup-001"]
    assert manifest["evaluation_type"] == "targeted_agent_check"
    report = (output / "baseline_report.md").read_text()
    assert "--case-id D008" in report and "--case-id agent-followup-001" in report


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
        "find_documents",
        "search_document",

        "get_chunk_context",
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
    async def fake_execute(*, spec, tool_args, **kwargs):
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
        spec = next((item for item in get_local_tool_specs(get_settings()) if item.name == "find_documents"), None)
        assert spec is not None
        asyncio.run(framework_tools.execute_project_tool(
            spec=spec,
            tool_args={},
        ))

    assert records == [
        {
            "tool_name": "find_documents",
            "status": "success",
            "ok": True,
            "output_keys": ["count", "private_value"],
            "tool_call_id": "",
            "has_evidence": False,
            "kb_chunk_ids": [],
            "kb_evidence": [],
        }
    ]
    assert "must-not-persist" not in repr(records)


@pytest.mark.parametrize("tool,ok,body,expected_count", [
    ("search_document", True, "private body must not persist", 1),
    ("search_document", False, "private body must not persist", 0),
    ("search_document", True, "", 0),
    ("find_documents", True, "private body must not persist", 0),
])
def test_capture_retains_only_successful_body_identity(tool, ok, body, expected_count, monkeypatch):
    async def execute(**kwargs):
        return {"ok": ok, "status": "success" if ok else "error", "tool_call_id": "call",
                "output": {"evidence": [{"id": "e1", "page_content": body, "metadata": {
                    "source_type": "knowledge_base", "file_name": "rules.md", "chunk_id": 1,
                    "source_path": "/kb/rules.md", "private_key": "must-not-persist",
                }}]}}
    monkeypatch.setattr(framework_tools, "execute_project_tool", execute)
    spec = next(item for item in get_local_tool_specs(get_settings()) if item.name == tool)
    with agent_evaluation.capture_tool_output_keys() as records:
        asyncio.run(framework_tools.execute_project_tool(spec=spec, tool_args={}))
    assert len(records[0]["kb_evidence"]) == expected_count
    assert "private body" not in repr(records) and "must-not-persist" not in repr(records)
    if expected_count:
        assert records[0]["kb_evidence"] == [{
            "reference_id": "e1", "file_name": "rules.md", "source_path": "/kb/rules.md",
        }]


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
    )
    second = agent_evaluation.execute_agent_case(
        dataset.cases[0],
        settings=settings,
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
            "prediction_schema_version": 3,
            "retrieval_metric_k": 3,
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


@pytest.mark.parametrize("all_failed", [False, True])
def test_runner_completes_reports_with_runtime_failures(tmp_path, monkeypatch, all_failed):
    from app.services.chat_models import ChatServiceOutcome
    from scripts.evaluation import rag_evaluator, rag_predictions
    from scripts.evaluation.rag.common import ChunkIndex

    output = tmp_path / "run"
    eval_file = tmp_path / "questions.json"
    eval_file.write_text(json.dumps([{
        "qid": f"q{i}", "question": f"问题{i}", "category": "test", "difficulty": "easy",
        "expected_behavior": "answer", "gold_chunk_ids": [1], "gold_doc_ids": ["rules.md"],
    } for i in (1, 2, 3)]))
    attempted = []

    def chat_turn(request, **kwargs):
        attempted.append(request.question)
        failed = all_failed or request.question == "问题2"
        return ChatServiceOutcome(http_status=200, code=200, message="test", question=request.question, result={
            "status": "MODEL_ERROR" if failed else "OK",
            "answer": "unsafe" if failed else "正常回答", "reason": "model failure" if failed else "",
        })

    monkeypatch.setattr(rag_predictions, "initialize_rag_for_predictions", lambda **k: None)
    monkeypatch.setattr(rag_predictions, "run_evaluation_chat_turn", chat_turn)
    monkeypatch.setattr(rag_evaluator, "load_chunk_index", lambda **k: ChunkIndex({1: "rules.md"}, {1: "政策"}, "test"))
    monkeypatch.setattr(runner, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(runner, "get_git_state", lambda: _fake_manifest("test")["code"])
    monkeypatch.setattr(runner, "build_manifest", lambda **kwargs: _fake_manifest(kwargs["run_id"]))
    monkeypatch.setattr("app.rag.pipeline.get_rag_status", lambda: {})

    assert runner.main([
        "--rag-only", "--agent-cases", str(FIXTURE), "--rag-eval-file", str(eval_file),
        "--output-dir", str(output),
    ]) == 0

    assert attempted == ["问题1", "问题2", "问题3"]
    failure_count = 3 if all_failed else 1
    with (output / "rag/predictions.csv").open(encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == rag_predictions.PREDICTION_FIELDNAMES
        assert len(list(reader)) == 3 - failure_count
    summary = json.loads((output / "baseline_summary.json").read_text())
    assert summary["rag"]["prediction_failure_count"] == failure_count
    assert summary["rag"]["prediction_failure_status_counts"] == {"MODEL_ERROR": failure_count}
    assert summary["rag"]["overall"]["prediction_coverage"] == (0 if all_failed else 0.6667)
    assert summary["rag"]["category:test"]["prediction_coverage"] == (0 if all_failed else 0.6667)
    if all_failed:
        assert summary["rag"]["overall"]["avg_final_score"] == 0
    failures = [json.loads(line) for line in (output / "failure_cases.jsonl").read_text().splitlines()]
    runtime_failures = [row for row in failures if row["failure_reasons"] == ["runtime_error"]]
    assert len(runtime_failures) == failure_count
    sidecar = [json.loads(line) for line in (output / "rag/prediction_failures.jsonl").read_text().splitlines()]
    assert {row["qid"] for row in sidecar} == {row["case_id"] for row in runtime_failures}
    assert (output / "manifest.json").exists()
    assert (output / "baseline_report.md").exists()
    assert (output / "rag/per_question_report.csv").exists()
    assert (output / "rag/summary_report.json").exists()
    for path in output.rglob("*"):
        if path.is_file():
            assert "unsafe" not in path.read_text(encoding="utf-8-sig")


def test_runner_writes_expected_output_structure(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    output = tmp_path / "run"
    rag_call = {}
    agent_call = {}

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
        lambda *args, **kwargs: (agent_call.update(kwargs) or [], agent_summary),
    )
    def fake_rag_baseline(**kwargs):
        rag_call.update(kwargs)
        return {
            "rows": [],
            "summary": {"overall": {"prediction_coverage": 1.0}},
            "judge_enabled": False,
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
            "--retrieval-k",
            "3",
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
    assert "Raw scores are not strictly comparable" in report
    assert "Context metrics: unavailable / not applicable" in report
    assert rag_call["retrieval_k"] == 3
    assert "top_k" not in agent_call
    assert "decompose_top_k" not in agent_call


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
