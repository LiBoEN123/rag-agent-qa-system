"""Extended Tool Evaluation 的安全 catalog、preflight 与 Secret 扫描。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.config import get_settings
from app.config import AppSettings
from app.tools.mcp.config import load_mcp_config
from app.tools.catalog import get_local_tool_specs
from scripts.evaluation.contracts import AgentEvaluationDataset


MCP_SUITES = frozenset({"mcp", "mixed"})
WEB_SUITES = frozenset({"online", "mixed"})
_TEXT_ARTIFACT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md"})


class McpPreflightError(RuntimeError):
    """表示数据集要求与真实安全 MCP catalog 不一致。"""

    reason = "REQUIRED_MCP_TOOL_NOT_AVAILABLE"


def safe_mcp_catalog_rows(
    catalog: Iterable[Mapping[str, Any]],
) -> list[dict[str, str | bool]]:
    """只保留允许写入评测产物的 MCP catalog 字段。"""
    rows = [
        {
            "server": str(item.get("server") or ""),
            "tool_name": str(item.get("tool_name") or ""),
            "remote_tool": str(item.get("remote_tool") or ""),
            "risk_level": str(item.get("risk_level") or "read_only"),
            "approval_required": bool(
                item.get("approval_required", False)
            ),
            "enabled": bool(item.get("enabled", False)),
            "enabled_for_default_principal": bool(
                item.get("enabled_for_default_principal", False)
            ),
        }
        for item in catalog
    ]
    return sorted(
        rows,
        key=lambda item: (str(item["server"]), str(item["tool_name"])),
    )


def build_tool_origin_catalog(
    mcp_catalog: Iterable[Mapping[str, Any]] = (),
) -> dict[str, dict[str, str | bool]]:
    """从本地 Registry 和 MCP Provider 构造名称到来源的安全映射。"""
    catalog: dict[str, dict[str, str | bool]] = {}
    registry = get_local_tool_specs(get_settings())
    for name in [item.name for item in registry]:
        spec = next((item for item in registry if item.name == name), None)
        if spec is None:
            continue
        origin = (
            "web"
            if any(
                permission.startswith("web:")
                for permission in spec.required_permissions
            )
            else "local"
        )
        catalog[name] = {
            "tool_name": name,
            "origin": origin,
            "server": "",
            "remote_tool": "",
            "risk_level": "read_only",
            "approval_required": False,
        }
    for row in safe_mcp_catalog_rows(mcp_catalog):
        name = str(row["tool_name"])
        if name in catalog:
            raise McpPreflightError(
                f"MCP tool 与本地 Registry 名称冲突: {name}"
            )
        catalog[name] = {
            **row,
            "origin": "mcp",
        }
    return catalog


def selected_mcp_servers(
    dataset: AgentEvaluationDataset,
    selected_suites: set[str] | frozenset[str],
) -> list[str]:
    """返回被选中且启用的 case 明确声明的 MCP Server。"""
    return sorted(
        {
            alias
            for case in dataset.cases
            if not case.disabled and case.suite in selected_suites
            for alias in case.mcp_servers
        }
    )


def validate_mcp_runtime_catalog(
    dataset: AgentEvaluationDataset,
    catalog: Iterable[Mapping[str, Any]],
    *,
    selected_suites: set[str] | frozenset[str],
) -> None:
    """在 discovery 后核对被选 case 的 Server 和 MCP expected tools。"""
    safe_rows = safe_mcp_catalog_rows(catalog)
    by_name = {str(row["tool_name"]): row for row in safe_rows}
    available_servers = {str(row["server"]) for row in safe_rows}
    required_servers = selected_mcp_servers(dataset, selected_suites)
    missing_servers = sorted(set(required_servers) - available_servers)
    failures: list[str] = []
    if missing_servers:
        failures.append("servers=" + ",".join(missing_servers))

    for case in dataset.cases:
        if case.disabled or case.suite not in selected_suites:
            continue
        for expected in case.iter_expected():
            if expected.tool_origin != "mcp":
                continue
            required_names = {
                expected.first_tool,
                *expected.tools_contains,
                *expected.tool_sequence,
            }
            required_names.discard("")
            missing_names = sorted(required_names - set(by_name))
            wrong_server = sorted(
                name
                for name in required_names & set(by_name)
                if by_name[name]["server"] != expected.mcp_server
            )
            if expected.first_tool_any_of and not any(
                name in by_name
                and by_name[name]["server"] == expected.mcp_server
                for name in expected.first_tool_any_of
            ):
                missing_names.append(
                    "any_of(" + ",".join(expected.first_tool_any_of) + ")"
                )
            if missing_names:
                failures.append(
                    f"{case.id}:tools=" + ",".join(sorted(missing_names))
                )
            if wrong_server:
                failures.append(
                    f"{case.id}:server_mismatch="
                    + ",".join(wrong_server)
                )
    if failures:
        raise McpPreflightError(
            f"{McpPreflightError.reason}: " + "; ".join(failures)
        )


def read_mcp_secret_values(settings: AppSettings) -> tuple[str, ...]:
    """在内存中解析 MCP Header Secret；调用方只能使用数量和匹配结果。"""
    config = load_mcp_config(settings.mcp_servers_config_path)
    values: set[str] = set()
    for server in config.servers.values():
        values.update(
            secret.get_secret_value()
            for secret in server.headers.values()
            if secret.get_secret_value()
        )
        values.update(
            value
            for env_name in server.headers_env.values()
            if (value := os.environ.get(env_name))
        )
    return tuple(sorted(values))


def scan_artifacts_for_secret_values(
    output_dir: Path,
    secret_values: Iterable[str],
) -> dict[str, Any]:
    """逐值扫描文本评测产物；结果只包含计数和相对路径。"""
    values = tuple(dict.fromkeys(value for value in secret_values if value))
    artifacts = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in _TEXT_ARTIFACT_SUFFIXES
        and ".isolated" not in path.parts
    )
    leak_paths: list[str] = []
    for path in artifacts:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            continue
        if any(value in content for value in values):
            leak_paths.append(path.relative_to(output_dir).as_posix())
    return {
        "secret_values_scanned": len(values),
        "artifacts_scanned": len(artifacts),
        "secret_leaks_found": len(leak_paths),
        "leak_paths": leak_paths,
    }


__all__ = [
    "MCP_SUITES",
    "WEB_SUITES",
    "McpPreflightError",
    "build_tool_origin_catalog",
    "read_mcp_secret_values",
    "safe_mcp_catalog_rows",
    "scan_artifacts_for_secret_values",
    "selected_mcp_servers",
    "validate_mcp_runtime_catalog",
]
