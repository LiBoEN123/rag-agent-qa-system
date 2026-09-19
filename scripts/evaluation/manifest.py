"""Evaluation Baseline v2 的安全、可复现运行清单。

用途：记录代码、数据、工具合同、模型配置和 Git 状态，确保结果可追溯、可比较。
输入：评测 Settings、输入文件、运行参数，以及本地仓库和运行环境的只读元数据。
输出：不含密钥的 ``manifest.json`` 数据及相关 SHA-256 指纹。
边界：只读取 Git/文件/环境信息；敏感字段会被拒绝或脱敏，不修改仓库和运行配置。
"""

from __future__ import annotations

import hashlib
import ast
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from app.config import AppSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "cookie",
    "password",
    "secret",
)
AGENT_PROMPT_INPUT_FILES = (
    "app/prompts.py",
    "app/agent/query_rewrite.py",
    "app/tools/catalog.py",
)
PREDICTION_SCHEMA_VERSION = 3
CORE_EVALUATION_MCP_MODE = "core_disabled"


def sha256_file(path: str | Path | None) -> str:
    """流式计算文件 SHA-256；缺失路径返回空串。"""
    if path is None:
        return ""
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    digest = hashlib.sha256()
    with resolved.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """按稳定 JSON 编码计算合同哈希。"""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _git(*args: str) -> str:
    """读取本地 Git 信息，不修改仓库。"""
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_git_state() -> dict[str, Any]:
    """返回 commit、branch 和 dirty 状态。"""
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(status),
    }


def build_tool_contract(
    settings: AppSettings,
) -> tuple[list[str], list[dict[str, Any]], str]:
    """从 LangChain 工具 schema 与治理声明构造当前合同。"""
    from app.agent.framework_tools import get_local_tool_contracts

    specs = get_local_tool_contracts(settings)
    return [spec["name"] for spec in specs], specs, sha256_json(specs)


def read_registered_tool_names() -> list[str]:
    """从当前 本地工具 catalog 源码静态读取 ToolSpec 名称，无导入副作用。"""
    registry_path = PROJECT_ROOT / "app" / "tools" / "catalog.py"
    tree = ast.parse(
        registry_path.read_text(encoding="utf-8"),
        filename=str(registry_path),
    )
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if function_name != "ToolSpec":
            continue
        keyword = next(
            (item for item in node.keywords if item.arg == "name"),
            None,
        )
        if (
            keyword is not None
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            names.append(keyword.value.value)
    if not names or len(names) != len(set(names)):
        raise ValueError(
            "无法从当前 本地工具 catalog 静态解析唯一工具名合同"
        )
    return names


def _relative_path(path: Path) -> str:
    """尽量用仓库相对路径记录文件。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_index_metadata(
    settings: AppSettings,
    rag_status: dict[str, Any] | None,
) -> dict[str, Any]:
    """只读索引 manifest，并补充当前内存索引计数。"""
    manifest_path = settings.index_store_dir / "index_manifest.json"
    raw: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            raw = {}
    files = raw.get("files")
    files = files if isinstance(files, list) else []
    build_config = raw.get("build_config")
    build_config = build_config if isinstance(build_config, dict) else {}
    status = dict(rag_status or {})
    return {
        "index_manifest_path": _relative_path(manifest_path),
        "index_manifest_sha256": sha256_file(manifest_path),
        "index_source": str(status.get("index_source") or "unknown"),
        "document_count": int(status.get("doc_count") or len(files)),
        "chunk_count": int(status.get("chunk_count") or 0),
        "embedding_model": str(
            raw.get("embedding_model") or settings.embedding_model_name
        ),
        "rerank_model": str(
            raw.get("rerank_model") or settings.rerank_model_name
        ),
        "chunk_size": int(
            build_config.get("chunk_size", settings.chunk_size)
        ),
        "chunk_overlap": int(
            build_config.get("chunk_overlap", settings.chunk_overlap)
        ),
    }


def _prompt_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    """记录生产 Agent 的 prompt 源文件哈希。"""
    return {
        path: sha256_file(PROJECT_ROOT / path)
        for path in paths
        if (PROJECT_ROOT / path).is_file()
    }


def assert_manifest_safe(value: Any, path: str = "manifest") -> None:
    """递归拒绝敏感字段名、SecretStr 和不可序列化值。"""
    if isinstance(value, SecretStr):
        raise ValueError(f"{path} 不允许包含 SecretStr")
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"{path}.{key} 是禁止写入的敏感字段")
            assert_manifest_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_manifest_safe(item, f"{path}[{index}]")
    json.dumps(value, ensure_ascii=False, allow_nan=False)


def build_manifest(
    *,
    run_id: str,
    settings: AppSettings,
    production_mcp_tools_enabled: bool,
    agent_cases_path: Path | None,
    rag_eval_path: Path | None,
    coverage_report: dict[str, Any],
    retrieval_metric_k: int,
    include_online: bool,
    use_llm_judge: bool,
    judge_model: str,
    selected_suites: list[str] | tuple[str, ...] | None = None,
    mcp_catalog: list[dict[str, str | bool]] | None = None,
    selected_mcp_servers: list[str] | tuple[str, ...] = (),
    rag_status: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """用显式白名单生成 manifest，绝不 dump 完整 Settings。"""
    tool_names, _, contract_hash = build_tool_contract(settings)
    git_state = get_git_state()
    explicit_suites = selected_suites is not None
    suites = sorted(
        set(
            selected_suites
            or (["core", "online"] if include_online else ["core"])
        )
    )
    mcp_extended = bool({"mcp", "mixed"} & set(suites))
    web_enabled = bool({"online", "mixed"} & set(suites))
    external_network_enabled = bool(web_enabled or mcp_extended)
    if mcp_extended:
        evaluation_type = "extended_tool_evaluation"
    elif web_enabled:
        evaluation_type = "online_web_evaluation"
    else:
        evaluation_type = "core_baseline"
    safe_mcp_catalog = list(mcp_catalog or [])
    mcp_metadata: dict[str, Any] = {
        "production_enabled": bool(production_mcp_tools_enabled),
        "evaluation_enabled": bool(settings.mcp_tools_enabled),
        "mode": "extended_live" if mcp_extended else CORE_EVALUATION_MCP_MODE,
    }
    if explicit_suites:
        mcp_metadata.update(
            {
                "discovered_server_count": len(
                    {str(row.get("server") or "") for row in safe_mcp_catalog}
                ),
                "discovered_tool_count": len(safe_mcp_catalog),
                "selected_servers": sorted(set(selected_mcp_servers)),
                "tools": safe_mcp_catalog,
            }
        )
    agent_prompt_hashes = _prompt_hashes(AGENT_PROMPT_INPUT_FILES)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "evaluation_type": evaluation_type,
        "selected_suites": suites,
        "external_network_enabled": external_network_enabled,
        "web": {"enabled": web_enabled},
        "code": {
            **git_state,
            "python_version": platform.python_version(),
            "python_executable": Path(sys.executable).name,
            "platform": platform.platform(),
        },
        "data": {
            "agent_cases_path": (
                _relative_path(agent_cases_path)
                if agent_cases_path is not None
                else ""
            ),
            "agent_cases_sha256": sha256_file(agent_cases_path),
            "rag_eval_path": (
                _relative_path(rag_eval_path)
                if rag_eval_path is not None
                else ""
            ),
            "rag_eval_sha256": sha256_file(rag_eval_path),
            "case_count": int(coverage_report.get("active_cases", 0)),
            "suite_counts": dict(coverage_report.get("suite", {})),
            "category_counts": dict(
                coverage_report.get("category", {})
            ),
        },
        "rag": {
            **_read_index_metadata(settings, rag_status),
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "prediction_producer": "production_chat_service",
            "retrieval_metric_k": retrieval_metric_k,
            "tool_mode": "rag_only",
            "permissions": ["knowledge_base:read"],
        },
        "agent": {
            "answer_model": settings.model_name,
            "query_rewrite_enabled": settings.enable_query_rewrite,
            "query_rewrite_model": (
                settings.query_rewrite_model_name_resolved
            ),
            "temperature": settings.temperature,
            "max_tokens": settings.llm_max_tokens,
            "budget": {
                "max_iterations": settings.agent_max_iterations,
                "max_tool_calls": settings.agent_max_tool_calls,
                "max_tool_errors": settings.agent_max_tool_errors,
                "max_consecutive_same_tool_calls": (
                    settings.agent_max_consecutive_same_tool_calls
                ),
                "max_consecutive_same_tool_errors": (
                    settings.agent_max_consecutive_same_tool_errors
                ),
                "max_total_cost_ms": settings.agent_max_total_cost_ms,
            },
            "registered_tool_names": tool_names,
            "tool_contract_sha256": contract_hash,
            "include_online": include_online,
        },
        "mcp": mcp_metadata,
        "prompts": {
            "agent": {
                "files": agent_prompt_hashes,
                "combined_sha256": sha256_json(agent_prompt_hashes),
            },
        },
        "judge": {
            "enabled": use_llm_judge,
            "model": judge_model if use_llm_judge else "",
            "cache_source": "",
        },
    }
    assert_manifest_safe(manifest)
    return manifest


__all__ = [
    "assert_manifest_safe",
    "build_manifest",
    "build_tool_contract",
    "get_git_state",
    "read_registered_tool_names",
    "sha256_file",
    "sha256_json",
]
