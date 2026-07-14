"""Agent Runtime 选择入口。"""

from collections.abc import Callable
from typing import Any

from app.config import validate_agent_runtime


AgentRuntimeRunner = Callable[..., dict[str, Any]]


def select_agent_runtime(
    runtime_name: str | None,
    *,
    legacy_runner: AgentRuntimeRunner,
    langgraph_runner: AgentRuntimeRunner,
) -> AgentRuntimeRunner:
    """校验配置并返回唯一选中的 Runtime，不做静默降级。"""
    normalized = validate_agent_runtime(runtime_name)
    if normalized == "legacy":
        return legacy_runner
    if normalized == "langgraph":
        return langgraph_runner
    raise ValueError(f"不支持的 Agent Runtime: {normalized}")


__all__ = ["AgentRuntimeRunner", "select_agent_runtime"]
