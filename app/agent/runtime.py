"""唯一 Agent Runtime：准备请求、执行 ``create_agent`` 并返回 AgentResult。"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage
from pydantic import TypeAdapter, ValidationError

import app.agent.framework_tools as framework_tools
from app.agent.framework_agent import build_framework_agent
from app.agent.framework_middleware import ProjectToolGovernanceMiddleware
from app.agent.results import (
    build_agent_runtime_error_result,
    build_framework_agent_result,
    build_short_circuit_result,
)
from app.agent.router import RouteDecision, route_agent_action
from app.common.logging import get_logger
from app.config import AppSettings, get_settings
from app.tools.context import ToolInvocationContext, build_default_tool_context
from app.tools.mcp.provider import list_mcp_tools_for_context


logger = get_logger("chat_service")

_INTEGER_ADAPTER = TypeAdapter(int)


class CompiledAgentRuntime(Protocol):
    """Runtime 和 Streaming 使用的 ``create_agent`` 可执行边界。"""

    # 约定编译后运行时的同步调用接口，显式传入配置和工具上下文。
    def invoke(
        self,
        input_value: Mapping[str, object],
        *,
        config: Mapping[str, object],
        context: ToolInvocationContext,
    ) -> object: ...

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


def validate_runtime_counts(
    *,
    top_k: object,
    decompose_top_k: object,
) -> tuple[int, int]:
    """保留公开 Runtime 数字输入原有的边界约束。"""

    # 将召回数量校验为指定范围内的整数，并拒绝布尔值。
    def validate(name: str, value: object, lower: int, upper: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        try:
            normalized = _INTEGER_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if normalized < lower or normalized > upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return normalized

    return (
        validate("top_k", top_k, 1, 10),
        validate("decompose_top_k", decompose_top_k, 1, 20),
    )


@lru_cache(maxsize=4)
def _get_cached_framework_agent(
    settings: AppSettings,
) -> CompiledAgentRuntime:
    """缓存不依赖 MCP discovery 的本地 ``create_agent`` 结构。"""
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
    *,
    tool_context: ToolInvocationContext | None = None,
) -> CompiledAgentRuntime:
    """Local 权限变化共用编译缓存；仅额外 MCP 工具集合需要构图。"""
    resolved_settings = settings or get_settings()
    if tool_context is None or not list_mcp_tools_for_context(tool_context):
        return _get_cached_framework_agent(resolved_settings)

    governance = ProjectToolGovernanceMiddleware(settings=resolved_settings)
    return cast(
        CompiledAgentRuntime,
        build_framework_agent(
            resolved_settings,
            tools=framework_tools.resolve_agent_tools(
                tool_context,
                resolved_settings,
            ),
            governance=governance,
        ),
    )


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
    top_k: int
    decompose_top_k: int
    trace_id: str
    session_id: str | None


def prepare_agent_runtime_invocation(
    *,
    top_k: int,
    decompose_top_k: int,
    trace_id: str,
    session_id: str | None,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
    settings: AppSettings | None = None,
) -> AgentRuntimeInvocation:
    """用普通 Python 完成参数校验、Router 与 Agent 调用准备。"""
    resolved_settings = settings or get_settings()
    resolved_top_k, resolved_decompose_top_k = validate_runtime_counts(
        top_k=top_k,
        decompose_top_k=decompose_top_k,
    )
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
            top_k=resolved_top_k,
            decompose_top_k=resolved_decompose_top_k,
            settings=resolved_settings,
        )
    else:
        resolved_context = replace(
            tool_context,
            effective_question=resolved_effective,
            top_k=resolved_top_k,
            decompose_top_k=resolved_decompose_top_k,
        )
    agent = (
        get_compiled_agent_runtime(
            resolved_settings,
            tool_context=resolved_context,
        )
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
        top_k=resolved_top_k,
        decompose_top_k=resolved_decompose_top_k,
        trace_id=trace_id,
        session_id=session_id,
    )


def run_agent(
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    top_k: int,
    decompose_top_k: int,
    trace_id: str = "",
    session_id: str | None = None,
    bad_case: dict[str, Any] | None = None,
    *,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
) -> dict[str, Any]:
    """路由后直接执行 ``create_agent`` 并返回已验证 Result。"""
    resolved_original = (original_question or "").strip()
    resolved_effective = (effective_question or "").strip()
    try:
        invocation = prepare_agent_runtime_invocation(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original,
            effective_question=resolved_effective,
            tool_context=tool_context,
        )
        logger.info(
            "[trace_id=%s] [agent] runtime_start action=%s session_id_present=%s",
            trace_id,
            invocation.decision.get("action", ""),
            bool((session_id or "").strip()),
        )
        if invocation.agent is None:
            result = build_short_circuit_result(invocation)
        else:
            output = invocation.agent.invoke(
                invocation.agent_input,
                config=invocation.agent_config,
                context=invocation.tool_context,
            )
            if not isinstance(output, Mapping):
                raise RuntimeError("create_agent did not produce a valid state")
            result = build_framework_agent_result(invocation, output)
        logger.info(
            "[trace_id=%s] [agent] runtime_end status=%s",
            trace_id,
            result.get("status", ""),
        )
        return result
    except Exception as exc:
        logger.exception("[trace_id=%s] [agent] runtime_failed", trace_id)
        return build_agent_runtime_error_result(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original,
            effective_question=resolved_effective,
            exc=exc,
        )


__all__ = [
    "AgentRuntimeInvocation",
    "CompiledAgentRuntime",
    "get_compiled_agent_runtime",
    "prepare_agent_runtime_invocation",
    "reset_agent_runtime",
    "run_agent",
    "validate_runtime_counts",
]
