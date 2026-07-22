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
from app.tools.context import ToolInvocationContext  # noqa: E402
from app.tools.contracts import ToolSpec  # noqa: E402
from app.tools.providers import ToolProvider, get_default_tool_provider  # noqa: E402


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


def _tool_data(provider: ToolProvider) -> tuple[list[dict[str, Any]], list[str]]:
    """Read registered tool contracts and model-facing schemas without dispatch."""
    context = ToolInvocationContext(source="agent_runtime_inspection")
    registry = getattr(provider, "registry", None)
    specs = list(provider.list_specs(context=context))
    tools_by_name = {
        tool.name: tool for tool in build_framework_tools(provider, context=context)
    }
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec):
            errors.append(f"provider returned non-ToolSpec: {type(spec).__name__}")
            continue
        if spec.name in seen:
            errors.append(f"duplicate tool name: {spec.name}")
        seen.add(spec.name)
        if not spec.description or not spec.version or spec.args_model is None:
            errors.append(f"incomplete ToolSpec contract: {spec.name}")
        if registry is None or registry.get(spec.name) is not spec:
            errors.append(f"registry cannot resolve ToolSpec: {spec.name}")
        tool = tools_by_name.get(spec.name)
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
                "requires_approval": spec.requires_approval,
            }
        )
    return sorted(rows, key=lambda item: item["name"]), errors


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
    tools, tool_errors = _tool_data(provider or get_default_tool_provider())
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
    checks = [
        _check("dependency_versions", not missing_packages, ", ".join(missing_packages)),
        _check("parent_required_nodes", not missing_parent, ", ".join(missing_parent)),
        _check("parent_forbidden_nodes", not forbidden_parent, ", ".join(forbidden_parent)),
        _check("framework_core_nodes", not missing_framework, ", ".join(missing_framework)),
        _check("framework_call_limits", not missing_limits, ", ".join(missing_limits)),
        _check("parent_xray_subgraph", xray_has_agent, "framework model/tools visible"),
        _check("default_tool_contracts", not tool_errors and bool(tools), "; ".join(tool_errors)),
        _check("source_file_budgets", not over_budget, "; ".join(over_budget)),
    ]
    return {
        "versions": versions,
        "parent_graph": {**parent_data, "xray_nodes": xray_nodes},
        "framework_agent": framework_data,
        "middleware_nodes": sorted(
            node for node in framework_data["nodes"] if "Middleware" in node
        ),
        "tools": tools,
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
    lines.extend(["", "Default tools"])
    lines.extend(f"- {tool['name']}" for tool in report["tools"])
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
