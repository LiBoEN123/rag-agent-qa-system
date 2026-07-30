"""统一运行可复现的 Evaluation Baseline v2。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppSettings, get_settings  # noqa: E402
from scripts.evaluation_contracts import (  # noqa: E402
    AgentEvaluationCase,
    AgentEvaluationDataset,
    build_coverage_report,
    load_agent_evaluation_dataset,
    validate_tool_names,
)
from scripts.evaluation_manifest import (  # noqa: E402
    build_manifest,
    get_git_state,
    read_registered_tool_names,
)
from scripts.evaluation_report import (  # noqa: E402
    build_markdown_report,
    classify_rag_failures,
    write_agent_csv,
    write_json,
    write_jsonl,
)


DEFAULT_AGENT_CASES = (
    PROJECT_ROOT / "data" / "eval" / "agent_eval_cases_v2.json"
)
DEFAULT_RAG_EVAL = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "rag_eval_questions_v2_semantic_fixed.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation_baseline_v2"


class InMemorySessionAdapter:
    """评测专用 session 边界，不触碰 SQLite。"""

    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str | None) -> list[dict[str, str]]:
        """返回隔离副本。"""
        return [
            dict(item)
            for item in self._messages.get(str(session_id or ""), [])
        ]

    def append_message(
        self,
        session_id: str | None,
        role: str,
        content: str,
    ) -> None:
        """按生产顺序保存 user/assistant 消息。"""
        key = str(session_id or "")
        self._messages.setdefault(key, []).append(
            {"role": role, "content": content}
        )

    def clear(self) -> None:
        """清空当前 case 的全部历史。"""
        self._messages.clear()


def _clear_runtime_caches() -> None:
    """清理会持有 Settings 或副作用边界的进程内缓存。"""
    from app.graph.builder import reset_agent_graph_runtime
    from app.llm.factory import clear_model_factory_cache
    from app.tools.registry import clear_tool_registry_cache
    from app.tools.runtime import close_tool_runtime

    reset_agent_graph_runtime()
    close_tool_runtime()
    clear_tool_registry_cache()
    clear_model_factory_cache()


@contextmanager
def evaluation_settings_scope(
    settings: AppSettings,
) -> Iterator[None]:
    """让评测调用链使用无生产写入的 Settings 快照。"""
    import app.config as config_module

    original = config_module.get_settings

    def provider() -> AppSettings:
        return settings

    patched: list[tuple[ModuleType, Any]] = []
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("app.")
            and isinstance(module, ModuleType)
            and getattr(module, "get_settings", None) is original
        ):
            patched.append((module, original))
            setattr(module, "get_settings", provider)
    config_module.get_settings = provider
    _clear_runtime_caches()
    try:
        yield
    finally:
        _clear_runtime_caches()
        config_module.get_settings = original
        for name, module in list(sys.modules.items()):
            if (
                name.startswith("app.")
                and isinstance(module, ModuleType)
                and getattr(module, "get_settings", None) is provider
            ):
                setattr(module, "get_settings", original)


def build_evaluation_settings(
    settings: AppSettings,
    output_dir: Path,
) -> AppSettings:
    """从当前真实配置复制只读评测快照。"""
    return settings.model_copy(
        update={
            "langgraph_checkpointer": "disabled",
            "enable_agent_trace_store": False,
            "tool_audit_enabled": False,
            "sqlite_db_path": output_dir / ".isolated" / "app.sqlite",
            "langgraph_checkpoint_db_path": (
                output_dir / ".isolated" / "checkpoints.sqlite3"
            ),
            "metrics_path": output_dir / ".isolated" / "metrics.jsonl",
            "bad_cases_path": output_dir / ".isolated" / "bad_cases.jsonl",
            "tool_audit_path": output_dir / ".isolated" / "tool_audit.jsonl",
        }
    )


def validate_dataset(
    path: Path,
    settings: AppSettings | None = None,
) -> tuple[
    AgentEvaluationDataset,
    dict[str, Any],
    list[str],
]:
    """只做加载、Schema、ID、工具合同和覆盖率检查。"""
    del settings
    dataset = load_agent_evaluation_dataset(path)
    tool_names = read_registered_tool_names()
    validate_tool_names(dataset, tool_names)
    coverage = build_coverage_report(dataset, tool_names)
    return dataset, coverage, tool_names


def _permissions_for_suite(suite: str) -> tuple[str, ...]:
    """Core 禁用在线权限，Online 才显式开放。"""
    if suite == "online":
        return ("knowledge_base:read", "web:search", "web:fetch")
    return ("knowledge_base:read",)


def _execute_agent_case(
    case: AgentEvaluationCase,
    *,
    settings: AppSettings,
    top_k: int,
    decompose_top_k: int,
) -> dict[str, Any]:
    """通过真实 ChatService 执行一个隔离 case。"""
    from dataclasses import replace

    from app.observability.metrics import (
        attach_query_rewrite_trace,
        build_query_rewrite_trace,
        collect_trace_metrics,
    )
    from app.graph.runtime import run_agent_langgraph
    from app.services.chat_models import ChatServiceRequest
    from app.services.chat_service import (
        ChatService,
        get_default_chat_service_dependencies,
    )
    from app.tools.context import PrincipalType, ToolInvocationContext
    from scripts.evaluation_agent import (
        combine_case_turns,
        score_agent_turn,
    )

    session = InMemorySessionAdapter()
    captured_metrics: list[dict[str, Any]] = []
    base = get_default_chat_service_dependencies()
    permissions = _permissions_for_suite(case.suite)

    def run_runtime(**kwargs: Any) -> dict[str, Any]:
        kwargs["tool_context"] = ToolInvocationContext(
            trace_id=str(kwargs.get("trace_id") or ""),
            session_id=str(kwargs.get("session_id") or ""),
            principal_id="evaluation_baseline_v2",
            principal_type=PrincipalType.SERVICE,
            permissions=permissions,
            source="evaluation_baseline_v2",
        )
        return run_agent_langgraph(**kwargs)

    dependencies = replace(
        base,
        settings_provider=lambda: settings,
        get_history=session.get_history,
        run_agent_langgraph=run_runtime,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        log_and_write_metrics=lambda item: captured_metrics.append(
            dict(item)
        ),
        write_bad_cases=lambda item: None,
        append_message=session.append_message,
    )
    service = ChatService(dependencies)
    session_id = (
        case.session_id.strip()
        or f"evaluation-{case.id}"
    )
    turn_rows: list[dict[str, Any]] = []
    for turn_index, (question, expected) in enumerate(
        zip(case.iter_questions(), case.iter_expected(), strict=True),
        start=1,
    ):
        trace_id = f"evaluation-{case.id}-turn-{turn_index}"
        started = time.perf_counter()
        outcome = service.handle(
            ChatServiceRequest(
                question=question,
                top_k=top_k,
                decompose_top_k=decompose_top_k,
                session_id=session_id,
                trace_id=trace_id,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = dict(outcome.result)
        latest_metrics = (
            captured_metrics[-1] if captured_metrics else {}
        )
        runtime_error = ""
        if result.get("status") == "INTERNAL_ERROR":
            runtime_error = str(result.get("reason") or "internal error")
        turn_rows.append(
            score_agent_turn(
                case_id=case.id,
                category=case.category,
                suite=case.suite,
                tags=case.tags,
                turn_index=turn_index,
                question=question,
                expected=expected,
                result=result,
                metrics=latest_metrics,
                latency_ms=elapsed_ms,
                runtime_error=runtime_error,
            )
        )
    session.clear()
    return combine_case_turns(case.id, turn_rows)


def run_agent_baseline(
    dataset: AgentEvaluationDataset,
    *,
    settings: AppSettings,
    include_online: bool,
    top_k: int,
    decompose_top_k: int,
    rag_already_initialized: bool,
    force_rebuild: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """运行 Core 以及显式启用的 Online Agent cases。"""
    from scripts.evaluation_agent import calculate_agent_metrics

    selected = [
        case
        for case in dataset.cases
        if not case.disabled
        and (case.suite == "core" or include_online)
    ]
    skipped_online = sum(
        not case.disabled and case.suite == "online"
        for case in dataset.cases
    )
    if include_online:
        skipped_online = 0
    if not rag_already_initialized:
        from scripts.agent_eval import init_rag_for_eval

        init_rag_for_eval(force_rebuild=force_rebuild)
    rows = [
        _execute_agent_case(
            case,
            settings=settings,
            top_k=top_k,
            decompose_top_k=decompose_top_k,
        )
        for case in selected
    ]
    metrics = calculate_agent_metrics(
        rows,
        total_cases=len(selected) + skipped_online,
        skipped_cases=skipped_online,
    )
    return rows, metrics


def run_rag_baseline(
    *,
    rag_eval_path: Path,
    output_dir: Path,
    top_k: int,
    force_rebuild: bool,
    use_llm_judge: bool,
    judge_model: str | None,
) -> dict[str, Any]:
    """复用 predictions 与语义 evaluator 的 Python 函数。"""
    from scripts.evaluate_rag_results_semantic_fixed import (
        run_rag_evaluation,
    )
    from scripts.predictions import run_predictions

    rag_dir = output_dir / "rag"
    predictions_path = rag_dir / "predictions.csv"
    run_predictions(
        eval_file=rag_eval_path,
        out_file=predictions_path,
        top_k=top_k,
        force_rebuild=force_rebuild,
        fail_on_model_error=True,
    )
    return run_rag_evaluation(
        eval_file=rag_eval_path,
        pred_file=predictions_path,
        out_dir=rag_dir,
        k=top_k,
        use_llm_judge=use_llm_judge,
        judge_model=judge_model,
    )


def _resolve_output_dir(
    requested: Path | None,
    run_name: str | None,
) -> tuple[str, Path]:
    """解析 run ID 和单次运行目录。"""
    default_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (run_name or "").strip() or (
        requested.name if requested is not None else default_id
    )
    if requested is not None:
        return run_id, requested
    return run_id, DEFAULT_OUTPUT_ROOT / run_id


def _reproduce_command(args: argparse.Namespace, output_dir: Path) -> str:
    """生成不包含凭据的准确复现命令。"""
    parts = [
        "python scripts/run_evaluation_baseline.py",
        f"--agent-cases {args.agent_cases}",
        f"--rag-eval-file {args.rag_eval_file}",
        f"--output-dir {output_dir}",
        f"--top-k {args.top_k}",
        f"--decompose-top-k {args.decompose_top_k}",
    ]
    if args.agent_only:
        parts.append("--agent-only")
    if args.rag_only:
        parts.append("--rag-only")
    if args.include_online:
        parts.append("--include-online")
    if args.use_llm_judge:
        parts.append("--use-llm-judge")
    if args.judge_model:
        parts.append(f"--judge-model {args.judge_model}")
    if args.allow_dirty:
        parts.append("--allow-dirty")
    if args.force_rebuild:
        parts.append("--force-rebuild")
    return " \\\n  ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。"""
    parser = argparse.ArgumentParser(
        description="Evaluation Baseline v2",
    )
    parser.add_argument(
        "--agent-cases",
        type=Path,
        default=DEFAULT_AGENT_CASES,
    )
    parser.add_argument(
        "--rag-eval-file",
        type=Path,
        default=DEFAULT_RAG_EVAL,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--decompose-top-k", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent-only", action="store_true")
    mode.add_argument("--rag-only", action="store_true")
    parser.add_argument("--include-online", action="store_true")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行校验或完整 baseline。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    decompose_top_k = (
        args.decompose_top_k
        if args.decompose_top_k is not None
        else settings.decompose_top_k
    )
    args.decompose_top_k = decompose_top_k
    dataset, coverage, tool_names = validate_dataset(
        args.agent_cases,
        settings,
    )
    if args.validate_only:
        print("Evaluation dataset validation passed.")
        print(
            json.dumps(
                coverage,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0

    if args.top_k < 1 or decompose_top_k < 1:
        parser.error("top-k 与 decompose-top-k 必须大于 0")
    git_state = get_git_state()
    if git_state["git_dirty"] and not args.allow_dirty:
        parser.error(
            "工作区不干净，拒绝生成正式 baseline；"
            "实验运行请显式使用 --allow-dirty"
        )
    run_id, output_dir = _resolve_output_dir(
        args.output_dir,
        args.run_name,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"输出目录已存在且非空：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_settings = build_evaluation_settings(settings, output_dir)

    agent_rows: list[dict[str, Any]] = []
    rag_rows: list[dict[str, Any]] = []
    if args.rag_only:
        agent_summary: dict[str, Any] = {
            "skipped": True,
            "reason": "--rag-only",
        }
    else:
        agent_summary = {}
    if args.agent_only:
        rag_summary: dict[str, Any] = {
            "skipped": True,
            "reason": "--agent-only",
        }
    else:
        rag_summary = {}

    with evaluation_settings_scope(evaluation_settings):
        rag_result: dict[str, Any] | None = None
        if not args.agent_only:
            rag_result = run_rag_baseline(
                rag_eval_path=args.rag_eval_file,
                output_dir=output_dir,
                top_k=args.top_k,
                force_rebuild=args.force_rebuild,
                use_llm_judge=args.use_llm_judge,
                judge_model=args.judge_model,
            )
            rag_rows = rag_result["rows"]
            rag_summary = {
                **rag_result["summary"],
                "metrics_source": (
                    "scripts/rag_eval/evaluation.py"
                ),
                "judge_enabled": rag_result["judge_enabled"],
            }
        if not args.rag_only:
            agent_rows, agent_summary = run_agent_baseline(
                dataset,
                settings=evaluation_settings,
                include_online=args.include_online,
                top_k=args.top_k,
                decompose_top_k=decompose_top_k,
                rag_already_initialized=rag_result is not None,
                force_rebuild=(
                    args.force_rebuild and rag_result is None
                ),
            )

    from app.rag import get_rag_status

    rag_status = get_rag_status()
    manifest = build_manifest(
        run_id=run_id,
        settings=evaluation_settings,
        agent_cases_path=(
            None if args.rag_only else args.agent_cases
        ),
        rag_eval_path=(
            None if args.agent_only else args.rag_eval_file
        ),
        coverage_report=coverage,
        top_k=args.top_k,
        decompose_top_k=decompose_top_k,
        include_online=args.include_online,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model or settings.judge_model,
        rag_status=rag_status,
    )
    if manifest["code"]["git_dirty"] and not args.allow_dirty:
        parser.error("运行期间工作区变脏，拒绝写入正式 baseline")

    agent_failures = [
        {
            "kind": "agent",
            **row,
        }
        for row in agent_rows
        if not row["passed"]
    ]
    rag_failures = classify_rag_failures(rag_rows)
    failures = agent_failures + rag_failures
    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "git_commit": manifest["code"]["git_commit"],
        "git_dirty": manifest["code"]["git_dirty"],
        "agent": agent_summary,
        "rag": rag_summary,
        "failure_case_count": len(failures),
        "skipped": {
            "online": (
                not args.include_online
                and bool(coverage["suite"].get("online"))
            ),
            "llm_judge": not args.use_llm_judge,
            "agent": bool(agent_summary.get("skipped")),
            "rag": bool(rag_summary.get("skipped")),
        },
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "baseline_summary.json", summary)
    write_json(output_dir / "coverage_report.json", coverage)
    write_jsonl(output_dir / "failure_cases.jsonl", failures)
    if not args.rag_only:
        write_jsonl(output_dir / "agent" / "per_case.jsonl", agent_rows)
        write_agent_csv(output_dir / "agent" / "per_case.csv", agent_rows)
        write_json(output_dir / "agent" / "summary.json", agent_summary)
    reproduce = _reproduce_command(args, output_dir)
    report = build_markdown_report(
        manifest=manifest,
        coverage=coverage,
        summary=summary,
        failures=failures,
        reproduce_command=reproduce,
    )
    (output_dir / "baseline_report.md").write_text(
        report,
        encoding="utf-8",
    )
    print(f"Evaluation baseline written to: {output_dir}")
    print(f"Registered tools: {', '.join(tool_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
