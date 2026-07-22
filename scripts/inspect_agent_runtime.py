"""Inspect the Agent Runtime without invoking models, tools, or persistence."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

from app.agent.framework_agent import build_framework_agent  # noqa: E402
from app.agent.framework_tools import build_framework_tools  # noqa: E402
from app.graph.builder import build_agent_graph  # noqa: E402
from app.tools.context import (  # noqa: E402
    ToolInvocationContext,
    build_default_tool_context,
)
from app.tools.contracts import ToolSpec  # noqa: E402
from app.tools.providers import (  # noqa: E402
    LocalToolProvider,
    ToolProvider,
    get_default_tool_provider,
)
from app.tools.providers.base import is_tool_visible  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402


PACKAGES = ("langchain", "langchain-core", "langchain-openai", "langgraph", "pydantic")
EXPECTED_PARENT_NODES = {
    "route", "initialize", "framework_agent", "adapt_framework", "plan",
    "execute", "rag_subgraph", "compose", "finalize",
}
FORBIDDEN_PARENT_NODES = {"agent_model", "governed_tools", "limit_handler"}
SOURCE_FILE_LIMITS = {
    "app/graph/nodes.py": 350,
    "app/agent/framework_agent.py": 300,
    "app/agent/framework_middleware.py": 350,
    "app/agent/framework_tools.py": 300,
    "app/agent/guardrails.py": 150,
}


def _line_count(relative_path: str) -> int:
    """Count logical lines while ignoring trailing empty lines and line endings."""
    lines = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return len(lines)


def _graph_data(drawable_graph) -> dict[str, Any]:
    """Serialize the stable public fields exposed by LangGraph's drawable graph."""
    nodes = sorted(name for name in drawable_graph.nodes if not name.startswith("__"))
    edges = sorted(
        (
            {
                "source": edge.source,
                "target": edge.target,
                "conditional": bool(edge.conditional),
            }
            for edge in drawable_graph.edges
        ),
        key=lambda item: (item["source"], item["target"], item["conditional"]),
    )
    return {"nodes": nodes, "edges": edges}


def _tool_data(
    provider: ToolProvider,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
    dict[str, int],
    list[str],
]:
    """Read registered tool contracts and model-facing schemas without dispatch."""
    context = build_default_tool_context(source="agent_runtime_inspection")
    registry = getattr(provider, "registry", None)
    if not isinstance(registry, ToolRegistry):
        return [], [], ["provider does not expose a ToolRegistry"], {}, []
    visible_specs = list(provider.list_specs(context=context))
    visible_names = [spec.name for spec in visible_specs if isinstance(spec, ToolSpec)]
    registered_specs = [registry.get(name) for name in registry.names()]
    all_permissions = tuple(
        sorted(
            {
                permission
                for spec in registered_specs
                if isinstance(spec, ToolSpec)
                for permission in spec.required_permissions
            }
        )
    )
    inspection_context = ToolInvocationContext(
        principal_id="agent_runtime_inspection",
        permissions=all_permissions,
        source="agent_runtime_inspection",
    )
    all_tools_by_name = {
        tool.name: tool
        for tool in build_framework_tools(
            LocalToolProvider(registry),
            context=inspection_context,
        )
    }
    errors: list[str] = []
    visibility_errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    registered_names = set(registry.names())
    if not set(visible_names) <= registered_names:
        visibility_errors.append("default visible tools are not a registry subset")
    for spec in visible_specs:
        if not isinstance(spec, ToolSpec):
            visibility_errors.append(
                f"provider returned non-ToolSpec: {type(spec).__name__}"
            )
            continue
        if not is_tool_visible(spec, context=context):
            visibility_errors.append(
                f"visible tool lacks a required permission: {spec.name}"
            )
    for spec in registered_specs:
        if not isinstance(spec, ToolSpec):
            errors.append(f"registry returned non-ToolSpec: {type(spec).__name__}")
            continue
        if spec.name in seen:
            errors.append(f"duplicate tool name: {spec.name}")
        seen.add(spec.name)
        if not spec.description or not spec.version or spec.args_model is None:
            errors.append(f"incomplete ToolSpec contract: {spec.name}")
        if registry.get(spec.name) is not spec:
            errors.append(f"registry cannot resolve ToolSpec: {spec.name}")
        tool = all_tools_by_name.get(spec.name)
        if tool is None:
            errors.append(f"StructuredTool is missing: {spec.name}")
            continue
        try:
            schema = tool.tool_call_schema.model_json_schema()
        except Exception as exc:
            errors.append(f"invalid args schema for {spec.name}: {exc}")
            schema = {}
        description = " ".join(spec.description.split())
        rows.append(
            {
                "name": spec.name,
                "version": spec.version,
                "description": description if len(description) <= 240 else description[:237] + "...",
                "args_schema": schema,
                "risk_level": spec.risk_level.value,
                "required_permissions": list(spec.required_permissions),
                "requires_approval": spec.requires_approval,
            }
        )
    visibility = {
        "permission_count": len(context.permissions),
        "registered_tool_count": len(rows),
        "visible_tool_count": len(visible_names),
    }
    return (
        rows,
        visible_names,
        errors,
        visibility,
        visibility_errors,
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def inspect_runtime(provider: ToolProvider | None = None) -> dict[str, Any]:
    """Return a deterministic, JSON-safe snapshot of the current runtime."""
    versions: dict[str, str] = {}
    missing_packages: list[str] = []
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = ""
            missing_packages.append(package)

    parent = build_agent_graph().compile()
    parent_graph = parent.get_graph()
    parent_xray = parent.get_graph(xray=True)
    framework_graph = build_framework_agent().get_graph()
    parent_data = _graph_data(parent_graph)
    framework_data = _graph_data(framework_graph)
    xray_nodes = sorted(parent_xray.nodes)
    enforce_default_registry = provider is None
    resolved_provider = provider or get_default_tool_provider()
    (
        registered_tools,
        default_visible_tools,
        tool_errors,
        tool_visibility,
        visibility_errors,
    ) = _tool_data(
        resolved_provider
    )
    source_files = {
        path: {"lines": _line_count(path), "limit": limit}
        for path, limit in SOURCE_FILE_LIMITS.items()
    }

    parent_nodes = set(parent_data["nodes"])
    framework_nodes = set(framework_data["nodes"])
    missing_parent = sorted(EXPECTED_PARENT_NODES - parent_nodes)
    forbidden_parent = sorted(FORBIDDEN_PARENT_NODES & parent_nodes)
    missing_framework = sorted({"model", "tools"} - framework_nodes)
    missing_limits = [
        fragment
        for fragment in ("ModelCallLimitMiddleware", "ToolCallLimitMiddleware")
        if not any(fragment in node for node in framework_nodes)
    ]
    xray_has_agent = any(node.endswith(":model") for node in xray_nodes) and any(
        node.endswith(":tools") for node in xray_nodes
    )
    over_budget = [
        f"{path}: {item['lines']} > {item['limit']}"
        for path, item in source_files.items()
        if item["lines"] > item["limit"]
    ]
    web_spec = next(
        (
            item
            for item in registered_tools
            if item.get("name") == "web_search"
        ),
        None,
    )
    web_schema_fields = set(
        (web_spec or {}).get("args_schema", {}).get("properties", {})
    )
    web_contract_ok = bool(
        web_spec and web_spec.get("required_permissions") == ["web:search"]
    )
    default_has_web_permission = (
        "web:search"
        in build_default_tool_context(
            source="agent_runtime_inspection"
        ).permissions
    )
    web_default_visibility_ok = (
        ("web_search" in default_visible_tools) == default_has_web_permission
    )
    web_authorized_visibility_ok = True
    if enforce_default_registry:
        default_context = build_default_tool_context(
            source="agent_runtime_inspection"
        )
        authorized_context = ToolInvocationContext(
            principal_id=default_context.principal_id,
            principal_type=default_context.principal_type,
            permissions=tuple((*default_context.permissions, "web:search")),
            source="agent_runtime_inspection",
        )
        authorized_names = {
            spec.name
            for spec in resolved_provider.list_specs(context=authorized_context)
        }
        web_authorized_visibility_ok = "web_search" in authorized_names
    checks = [
        _check("dependency_versions", not missing_packages, ", ".join(missing_packages)),
        _check("parent_required_nodes", not missing_parent, ", ".join(missing_parent)),
        _check("parent_forbidden_nodes", not forbidden_parent, ", ".join(forbidden_parent)),
        _check("framework_core_nodes", not missing_framework, ", ".join(missing_framework)),
        _check("framework_call_limits", not missing_limits, ", ".join(missing_limits)),
        _check("parent_xray_subgraph", xray_has_agent, "framework model/tools visible"),
        _check(
            "registered_tool_contracts",
            not tool_errors and bool(registered_tools),
            "; ".join(tool_errors),
        ),
        _check(
            "default_tool_visibility",
            not visibility_errors,
            "; ".join(visibility_errors),
        ),
        _check(
            "web_search_contract",
            not enforce_default_registry or web_contract_ok,
            "web_search must require web:search",
        ),
        _check(
            "web_search_model_schema",
            not enforce_default_registry
            or not ({"proxy", "backend", "timeout", "runtime"} & web_schema_fields),
            "web_search model schema exposes runtime configuration",
        ),
        _check(
            "web_search_default_visibility",
            not enforce_default_registry or web_default_visibility_ok,
            "web_search visibility does not match default permissions",
        ),
        _check(
            "web_search_authorized_visibility",
            not enforce_default_registry or web_authorized_visibility_ok,
            "web_search is hidden despite web:search permission",
        ),
        _check("source_file_budgets", not over_budget, "; ".join(over_budget)),
    ]
    return {
        "versions": versions,
        "parent_graph": {**parent_data, "xray_nodes": xray_nodes},
        "framework_agent": framework_data,
        "middleware_nodes": sorted(
            node for node in framework_data["nodes"] if "Middleware" in node
        ),
        "registered_tools": registered_tools,
        "default_visible_tools": default_visible_tools,
        "tool_visibility": tool_visibility,
        "source_files": source_files,
        "checks": checks,
    }


def render_text(report: dict[str, Any]) -> str:
    """Render the compact maintainer-facing summary."""
    lines = ["Agent Runtime Inspection", "", "Framework versions"]
    lines.extend(f"- {name}: {value}" for name, value in report["versions"].items())
    lines.extend(["", "Parent graph nodes"])
    lines.extend(f"- {name}" for name in report["parent_graph"]["nodes"])
    lines.extend(["", "Framework agent nodes"])
    lines.extend(f"- {name}" for name in report["framework_agent"]["nodes"])
    lines.extend(["", "Registered tools"])
    for tool in report["registered_tools"]:
        permissions = ",".join(tool["required_permissions"]) or "none"
        lines.append(
            f"- {tool['name']} {tool['version']} risk={tool['risk_level']} "
            f"permissions={permissions} approval={tool['requires_approval']}"
        )
    lines.extend(["", "Default visible tools"])
    lines.extend(f"- {name}" for name in report["default_visible_tools"])
    visibility = report["tool_visibility"]
    lines.append(
        "- visibility: "
        f"{visibility['visible_tool_count']} visible / "
        f"{visibility['registered_tool_count']} registered / "
        f"{visibility['permission_count']} permissions"
    )
    lines.extend(["", "Architecture budgets"])
    lines.extend(
        f"- {Path(path).name}: {item['lines']} / {item['limit']}"
        for path, item in report["source_files"].items()
    )
    return "\n".join(lines)


def render_mermaid() -> str:
    """Delegate full xray Mermaid generation to LangGraph."""
    return build_agent_graph().compile().get_graph(xray=True).draw_mermaid()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json", "mermaid"), default="text")
    parser.add_argument("--check", action="store_true", help="run architecture checks")
    parser.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.format == "mermaid" and not args.check:
            print(render_mermaid().rstrip())
            return 0
        report = inspect_runtime()
        if args.check:
            failures = [item for item in report["checks"] if not item["passed"]]
            if failures:
                print("Agent runtime architecture checks failed:", file=sys.stderr)
                for failure in failures:
                    detail = failure["detail"] or "contract not satisfied"
                    print(f"- {failure['name']}: {detail}", file=sys.stderr)
                return 1
            print("Agent runtime architecture checks passed.")
            return 0
        if args.format == "json":
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(render_text(report))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"Agent runtime inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
