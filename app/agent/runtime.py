"""Agent Runtime 的参数校验、Router、上下文准备与构图缓存。"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage

import app.agent.framework_tools as framework_tools
from app.agent.framework_agent import build_framework_agent
from app.agent.framework_middleware import ProjectToolGovernanceMiddleware
from app.agent.router import RouteDecision, route_agent_action
from app.config import AppSettings, get_settings
from app.tools.context import ToolInvocationContext, build_default_tool_context


class CompiledAgentRuntime(Protocol):
    """Runtime 和 Streaming 使用的 ``create_agent`` 可执行边界。"""

    # 约定编译后运行时的异步事件流接口及流模式、子图和版本参数。
    def astream(
        self,
        input_value: Mapping[str, object],
        *,
        config: Mapping[str, object],
        context: ToolInvocationContext,
        stream_mode: list[str],
        subgraphs: bool,
        version: str,
    ) -> AsyncIterator[object]: ...

    # 约定获取运行时图结构的接口，供检查和可视化使用。
    def get_graph(self, **kwargs: Any) -> Any: ...


@lru_cache(maxsize=4)
def _get_cached_framework_agent(
    settings: AppSettings,
) -> CompiledAgentRuntime:
    """按配置缓存本地工具的 ``create_agent`` 结构。"""
    governance = ProjectToolGovernanceMiddleware(settings=settings)
    return cast(
        CompiledAgentRuntime,
        build_framework_agent(
            settings,
            tools=framework_tools.get_local_tools(settings),
            governance=governance,
        ),
    )


def get_compiled_agent_runtime(
    settings: AppSettings | None = None,
) -> CompiledAgentRuntime:
    """解析当前配置后读取缓存；请求权限与工具模式由 Middleware 处理。"""
    return _get_cached_framework_agent(settings or get_settings())


def reset_agent_runtime() -> None:
    """失效唯一 Agent Runtime 缓存，供配置切换和关闭流程使用。"""
    _get_cached_framework_agent.cache_clear()


@dataclass(frozen=True)
class AgentRuntimeInvocation:
    """一次直接 ``create_agent`` 调用的已解析强类型输入。"""

    agent: CompiledAgentRuntime | None
    agent_input: dict[str, list[HumanMessage]]
    agent_config: dict[str, int]
    tool_context: ToolInvocationContext
    decision: RouteDecision
    route_cost_ms: int
    original_question: str
    effective_question: str
    trace_id: str
    session_id: str | None


def prepare_agent_runtime_invocation(
    *,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
    settings: AppSettings | None = None,
) -> AgentRuntimeInvocation:
    """用普通 Python 完成参数校验、Router 与 Agent 调用准备。"""
    resolved_settings = settings or get_settings()
    resolved_original = (original_question or "").strip()
    resolved_effective = (effective_question or "").strip()
    route_started = time.perf_counter()
    decision = route_agent_action(resolved_effective)
    route_cost_ms = max(0, int((time.perf_counter() - route_started) * 1000))
    if tool_context is None:
        resolved_context = build_default_tool_context(
            trace_id=trace_id,
            session_id=session_id or "",
            source="chat_api",
            effective_question=resolved_effective,
            settings=resolved_settings,
        )
    else:
        resolved_context = replace(
            tool_context,
            effective_question=resolved_effective,
        )
    agent = (
        get_compiled_agent_runtime(resolved_settings)
        if decision.get("action") == "agent"
        else None
    )
    return AgentRuntimeInvocation(
        agent=agent,
        agent_input={"messages": [HumanMessage(content=resolved_effective)]},
        agent_config={
            "recursion_limit": max(
                25,
                resolved_settings.agent_max_iterations * 8 + 20,
            ),
            "max_concurrency": 1,
        },
        tool_context=resolved_context,
        decision=decision,
        route_cost_ms=route_cost_ms,
        original_question=resolved_original,
        effective_question=resolved_effective,
        trace_id=trace_id,
        session_id=session_id,
    )


__all__ = [
    "AgentRuntimeInvocation",
    "CompiledAgentRuntime",
    "get_compiled_agent_runtime",
    "prepare_agent_runtime_invocation",
    "reset_agent_runtime",
]
