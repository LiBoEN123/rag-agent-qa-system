"""统一运行可复现的 Evaluation Baseline v2。

用途：完成参数解析、离线合同校验、正式运行防护、Agent/RAG 调度与产物汇总。
输入：版本化评测集、suite/judge/索引选项、输出位置和当前 AppSettings。
输出：运行清单、逐题文件、汇总 JSON/Markdown，以及可用于自动化门禁的退出码。
边界：正式运行默认拒绝脏工作树；``--validate-only`` 不加载模型、RAG、网络或写结果目录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from app.config import AppSettings, get_settings
from scripts.evaluation.agent import run_agent_baseline
from scripts.evaluation.contracts import (
    AgentEvaluationDataset,
    build_coverage_report,
    load_agent_evaluation_dataset,
    validate_tool_names,
)
from scripts.evaluation.external import (
    MCP_SUITES,
    McpPreflightError,
    build_tool_origin_catalog,
    read_mcp_secret_values,
    safe_mcp_catalog_rows,
    scan_artifacts_for_secret_values,
    selected_mcp_servers,
    validate_mcp_runtime_catalog,
)
from scripts.evaluation.manifest import (
    build_manifest,
    get_git_state,
    read_registered_tool_names,
)
from scripts.evaluation.report import (
    build_markdown_report,
    classify_rag_failures,
    write_agent_csv,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


class McpRuntimeInitializationError(RuntimeError):
    """表示 Extended preflight 在 MCP runtime 初始化阶段失败。"""


def _clear_runtime_caches() -> None:
    """清理会持有 Settings 或副作用边界的进程内缓存。"""
    from app.agent.runtime import reset_agent_runtime
    from app.llm.factory import clear_model_factory_cache
    from app.tools.catalog import get_local_tool_specs
    from app.agent.framework_tools import get_local_tools
    from app.tools.mcp.client import reset_mcp_runtime

    reset_agent_runtime()
    get_local_tool_specs.cache_clear()
    get_local_tools.cache_clear()
    clear_model_factory_cache()
    reset_mcp_runtime()


@contextmanager
def evaluation_settings_scope(
    settings: AppSettings,
) -> Iterator[None]:
    """让评测调用链使用无生产写入的 Settings 快照。"""
    import app.config as config_module

    original = config_module.get_settings

    def provider() -> AppSettings:
        """向评测调用链返回当前隔离的 Settings 快照。"""
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
    *,
    mcp_enabled: bool = False,
) -> AppSettings:
    """复制无生产写入的评测快照；MCP 只由显式 Extended profile 开启。"""
    return settings.model_copy(
        update={
            "tool_audit_enabled": mcp_enabled,
            "mcp_tools_enabled": mcp_enabled,
            "sqlite_db_path": output_dir / ".isolated" / "app.sqlite",
            "metrics_path": output_dir / ".isolated" / "metrics.jsonl",
            "bad_cases_path": output_dir / ".isolated" / "bad_cases.jsonl",
            "tool_audit_path": output_dir / ".isolated" / "tool_audit.jsonl",
        }
    )


@contextmanager
def evaluation_runtime_scope(
    settings: AppSettings,
) -> Iterator[list[dict[str, str | bool]]]:
    """让 Settings scope 与 MCP initialize/close 生命周期严格配对。"""
    from app.tools.mcp.client import (
        close_mcp_runtime,
        initialize_mcp_runtime,
        reset_mcp_runtime,
    )
    from app.tools.mcp.provider import get_mcp_tool_catalog

    with evaluation_settings_scope(settings):
        initialized = False
        try:
            if settings.mcp_tools_enabled:
                try:
                    asyncio.run(initialize_mcp_runtime(settings))
                except Exception as exc:
                    raise McpRuntimeInitializationError(
                        "MCP_RUNTIME_INITIALIZATION_FAILED: "
                        f"{type(exc).__name__}"
                    ) from None
                initialized = True
                yield safe_mcp_catalog_rows(get_mcp_tool_catalog())
            else:
                reset_mcp_runtime()
                yield []
        finally:
            if initialized:
                asyncio.run(close_mcp_runtime())
            else:
                reset_mcp_runtime()


def _requested_suites(args: argparse.Namespace) -> set[str]:
    """保留 Core 默认语义，并为每类外部 suite 使用独立显式开关。"""
    suites = {"core"}
    if args.include_online:
        suites.add("online")
    if args.include_mcp:
        suites.add("mcp")
    if args.include_mixed:
        suites.add("mixed")
    return suites


def _selected_suites(
    dataset: AgentEvaluationDataset,
    args: argparse.Namespace,
) -> list[str]:
    """返回本次数据集中真正会执行的 suite。"""
    requested = _requested_suites(args)
    available = {
        case.suite for case in dataset.cases if not case.disabled
    }
    return sorted(requested & available)


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


def _summarize_prediction_failures(path: Path) -> dict[str, Any]:
    """从生产者完整写出的安全 sidecar 汇总 case runtime failure。"""
    with path.open(encoding="utf-8") as file:
        statuses = Counter(json.loads(line)["status"] for line in file if line.strip())
    return {
        "prediction_failure_count": sum(statuses.values()),
        "prediction_failure_status_counts": dict(statuses),
    }


def run_rag_baseline(
    *,
    rag_eval_path: Path,
    output_dir: Path,
    retrieval_k: int,
    force_rebuild: bool,
    use_llm_judge: bool,
    judge_model: str | None,
) -> dict[str, Any]:
    """生成生产 Agentic RAG 预测，K 仅作用于离线评分。"""
    from scripts.evaluation.rag_evaluator import run_rag_evaluation
    from scripts.evaluation.rag_predictions import PREDICTION_SCHEMA_VERSION, run_predictions

    rag_dir = output_dir / "rag"
    predictions_path = rag_dir / "predictions.csv"
    rows = run_predictions(
        eval_file=rag_eval_path, out_file=predictions_path, force_rebuild=force_rebuild,
    )
    if any(row.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION for row in rows):
        raise RuntimeError("RAG prediction requires Schema v3 records")
    failure_summary = _summarize_prediction_failures(rag_dir / "prediction_failures.jsonl")
    result = run_rag_evaluation(
        eval_file=rag_eval_path, pred_file=predictions_path, out_dir=rag_dir,
        k=retrieval_k, use_llm_judge=use_llm_judge, judge_model=judge_model,
    )
    result["summary"].update(failure_summary)
    return result


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
        f"--retrieval-k {args.retrieval_k}",
    ]
    if args.agent_only:
        parts.append("--agent-only")
    if args.rag_only:
        parts.append("--rag-only")
    if args.include_online:
        parts.append("--include-online")
    if args.include_mcp:
        parts.append("--include-mcp")
    if args.include_mixed:
        parts.append("--include-mixed")
    if args.use_llm_judge:
        parts.append("--use-llm-judge")
    if args.judge_model:
        parts.append(f"--judge-model {args.judge_model}")
    if args.allow_dirty:
        parts.append("--allow-dirty")
    if args.force_rebuild:
        parts.append("--force-rebuild")
    for case_id in args.case_ids:
        parts.append(f"--case-id {shlex.quote(case_id)}")
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
    parser.add_argument(
        "--retrieval-k", type=int, default=5,
        help="仅对 Agent 观察到的有序去重证据前 K 项评分，不控制工具 top_k",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--case-id", dest="case_ids", action="append", default=[],
        help="仅运行指定的完整 Agent case，可重复；真实运行须同时使用 --agent-only",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent-only", action="store_true")
    mode.add_argument("--rag-only", action="store_true")
    parser.add_argument("--include-online", action="store_true")
    parser.add_argument("--include-mcp", action="store_true")
    parser.add_argument("--include-mixed", action="store_true")
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
    if args.retrieval_k < 1:
        parser.error("retrieval-k 必须大于 0")
    dataset, coverage, tool_names = validate_dataset(
        args.agent_cases,
        settings,
    )
    if args.case_ids:
        if not args.agent_only and not args.validate_only:
            parser.error("--case-id 需要 --agent-only，避免意外运行完整 RAG Evaluation")
        if args.rag_only:
            parser.error("--case-id 不适用于 --rag-only")
        requested = set(args.case_ids)
        unknown = requested - {case.id for case in dataset.cases}
        if unknown:
            parser.error(f"unknown case id: {', '.join(sorted(unknown))}")
        selected_cases = [case for case in dataset.cases if case.id in requested]
        unavailable = [
            case.id for case in selected_cases
            if case.disabled or case.suite not in _requested_suites(args)
        ]
        if unavailable:
            parser.error(f"case 未启用（disabled 或 suite 开关缺失）: {', '.join(unavailable)}")
        dataset = dataset.model_copy(update={"cases": selected_cases})
        coverage = build_coverage_report(dataset, tool_names)
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

    selected_suites = _selected_suites(dataset, args)
    mcp_evaluation_enabled = bool(MCP_SUITES & set(selected_suites))

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
    evaluation_settings = build_evaluation_settings(
        settings,
        output_dir,
        mcp_enabled=mcp_evaluation_enabled,
    )

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

    mcp_catalog: list[dict[str, str | bool]] = []
    try:
        with evaluation_runtime_scope(evaluation_settings) as runtime_catalog:
            mcp_catalog = list(runtime_catalog)
            if mcp_evaluation_enabled:
                validate_mcp_runtime_catalog(
                    dataset,
                    mcp_catalog,
                    selected_suites=set(selected_suites),
                )
            tool_catalog = build_tool_origin_catalog(mcp_catalog)
            rag_result: dict[str, Any] | None = None
            if not args.agent_only:
                rag_result = run_rag_baseline(
                    rag_eval_path=args.rag_eval_file,
                    output_dir=output_dir,
                    retrieval_k=args.retrieval_k,
                    force_rebuild=args.force_rebuild,
                    use_llm_judge=args.use_llm_judge,
                    judge_model=args.judge_model,
                )
                rag_rows = rag_result["rows"]
                rag_summary = {
                    **rag_result["summary"],
                    "metrics_source": (
                        "scripts/evaluation/rag/evaluation.py"
                    ),
                    "judge_enabled": rag_result["judge_enabled"],
                }
            if not args.rag_only:
                agent_rows, agent_summary = run_agent_baseline(
                    dataset,
                    settings=evaluation_settings,
                    include_online=args.include_online,
                    include_mcp=args.include_mcp,
                    include_mixed=args.include_mixed,
                    rag_already_initialized=rag_result is not None,
                    force_rebuild=(
                        args.force_rebuild and rag_result is None
                    ),
                    tool_catalog=tool_catalog,
                )
    except (McpRuntimeInitializationError, McpPreflightError) as exc:
        write_json(
            output_dir / "preflight_failure.json",
            {
                "status": "PRECHECK_FAILED",
                "classification": "BLOCKED_EXTERNAL_ENVIRONMENT",
                "reason": str(exc),
            },
        )
        print(f"Extended MCP preflight failed: {exc}", file=sys.stderr)
        return 2

    from app.rag.runtime import get_rag_status

    rag_status = get_rag_status()
    manifest = build_manifest(
        run_id=run_id,
        settings=evaluation_settings,
        production_mcp_tools_enabled=settings.mcp_tools_enabled,
        agent_cases_path=(
            None if args.rag_only else args.agent_cases
        ),
        rag_eval_path=(
            None if args.agent_only else args.rag_eval_file
        ),
        coverage_report=coverage,
        retrieval_metric_k=args.retrieval_k,
        include_online=args.include_online,
        selected_suites=selected_suites,
        mcp_catalog=mcp_catalog,
        selected_mcp_servers=selected_mcp_servers(
            dataset,
            set(selected_suites),
        ),
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model or settings.judge_model,
        rag_status=rag_status,
    )
    if args.case_ids:
        manifest["evaluation_type"] = "targeted_agent_check"
        manifest["data"]["selected_case_ids"] = [case.id for case in dataset.cases]
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
            "mcp": "mcp" not in selected_suites,
            "mixed": "mixed" not in selected_suites,
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
    secret_values = (
        read_mcp_secret_values(evaluation_settings)
        if mcp_evaluation_enabled
        else ()
    )
    secret_scan = scan_artifacts_for_secret_values(
        output_dir,
        secret_values,
    )
    summary["secret_scan"] = {
        key: secret_scan[key]
        for key in (
            "secret_values_scanned",
            "artifacts_scanned",
            "secret_leaks_found",
        )
    }
    write_json(output_dir / "baseline_summary.json", summary)
    secret_scan = scan_artifacts_for_secret_values(
        output_dir,
        secret_values,
    )
    write_json(output_dir / "secret_scan.json", secret_scan)
    if secret_scan["secret_leaks_found"]:
        raise RuntimeError(
            "Extended Evaluation artifact Secret scan failed closed"
        )
    print(f"Evaluation baseline written to: {output_dir}")
    print(f"Registered tools: {', '.join(tool_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
