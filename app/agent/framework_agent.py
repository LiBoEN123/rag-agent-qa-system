"""构建 create_agent 子图并转换其最小业务事件。"""

import operator
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import AgentState as LangChainAgentState
from typing_extensions import NotRequired

from app.agent.framework_middleware import ProjectToolGovernanceMiddleware
from app.agent.framework_tools import RegistryToolsMiddleware
from app.agent.model import get_agent_chat_model
from app.agent.state import AgentObservation
from app.agent.tool_events import build_tool_observation, build_tool_trace
from app.config import AppSettings, get_settings
from app.tools.context import ToolInvocationContext


def _prefer_stop_reason(left: str, right: str) -> str:
    """归并并优先保留最新的非空停止原因。"""
    return right or left


class FrameworkAgentState(LangChainAgentState):
    """create_agent 与父图共享的最小治理状态。"""

    governance_events: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    agent_elapsed_ms: NotRequired[int]
    stop_reason: NotRequired[Annotated[str, _prefer_stop_reason]]
    pending_rag_result: NotRequired[dict[str, Any]]


class FrameworkEventProjection(TypedDict):
    """框架治理事件投影到父图业务字段后的纯更新。"""

    observations: list[AgentObservation]
    tool_trace: list[dict[str, Any]]
    framework_events_applied: int


class _DeferredAgentModel:
    """把模型实例解析延迟到请求执行，构图时不触发模型配置。"""

    profile: dict[str, Any] = {}

    @staticmethod
    def _model():
        """延迟取得当前请求使用的 Agent 模型。"""
        model = get_agent_chat_model()
        if model is None:
            raise RuntimeError("Agent model is unavailable")
        return model

    def bind_tools(self, tools, **kwargs):
        """把动态工具绑定转发给实际模型。"""
        clean_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        return self._model().bind_tools(tools, **clean_kwargs)

    def bind(self, **kwargs):
        """兼容转发模型的通用绑定参数。"""
        model = self._model()
        bind = getattr(model, "bind", None)
        return bind(**kwargs) if callable(bind) else model


def build_framework_agent(settings: AppSettings | None = None):
    """创建使用框架循环、动态工具和内置调用限制的 Agent 子图。"""
    resolved = settings or get_settings()
    agent = create_agent(
        _DeferredAgentModel(),
        tools=[],
        middleware=[
            RegistryToolsMiddleware(),
            ProjectToolGovernanceMiddleware(settings=resolved),
            ModelCallLimitMiddleware(
                run_limit=resolved.agent_max_iterations,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=resolved.agent_max_tool_calls,
                exit_behavior="continue",
            ),
        ],
        state_schema=FrameworkAgentState,
        context_schema=ToolInvocationContext,
        name="framework_agent",
    )
    # 当前 ToolNode 对同批调用使用 Send；顺序执行使 fatal fail-closed 可确定。
    return agent.with_config({"max_concurrency": 1})


def _model_trace(event: dict[str, Any], index: int) -> dict[str, Any]:
    """把模型治理事件转换为现有迭代 trace。"""
    return {
        "step": "agent_iteration",
        "iteration_index": index,
        "status": "success" if event.get("ok") else "fallback",
        "model_has_tool_calls": bool(event.get("has_tool_calls")),
        "requested_tool_call_count": int(
            event.get("requested_tool_call_count", 0) or 0
        ),
        "model_cost_ms": max(0, int(event.get("cost_ms", 0) or 0)),
        "available_tool_names": list(event.get("available_tool_names", []))[:50],
        "error_type": str(event.get("error_type") or ""),
    }


def _guardrail_trace(event: dict[str, Any]) -> dict[str, Any]:
    """为未执行的治理拦截生成兼容 trace。"""
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    return {
        "step": "tool_call",
        "action": "tool_call",
        "tool_name": str(event.get("name") or ""),
        "tool_status": "error",
        "success": False,
        "error": str(result.get("error") or "Agent governance stopped tool execution"),
        "error_code": str(event.get("error_code") or ""),
        "tool_call_id": str(event.get("tool_call_id") or ""),
        "cost_ms": 0,
        "handler_executed": False,
        "will_return_to_agent_model": False,
        "stop_reason": str(event.get("stop_reason") or ""),
        "call_fingerprint": str(event.get("call_fingerprint") or ""),
    }


def _project_tool_event(
    event: dict[str, Any],
    *,
    step_id: int,
) -> tuple[AgentObservation, dict[str, Any]]:
    """把已执行工具事件映射为 observation 和 tool trace。"""
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    tool_name = str(event.get("name") or "")
    tool_args = event.get("args") if isinstance(event.get("args"), dict) else {}
    observation = build_tool_observation(
        step_id=step_id,
        tool_name=tool_name,
        result=result,
        cost_ms=max(0, int(event.get("cost_ms", 0) or 0)),
    )
    trace = build_tool_trace(
        step_id=step_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
        observation=observation,
    )
    trace.update(
        {
            "failure_kind": str(event.get("failure_kind") or ""),
            "handler_executed": bool(event.get("handler_executed")),
            "will_return_to_agent_model": not bool(event.get("stop_reason")),
            "stop_reason": str(event.get("stop_reason") or ""),
            "call_fingerprint": str(event.get("call_fingerprint") or ""),
        }
    )
    return observation, trace


def project_framework_events(
    graph_state: Mapping[str, Any],
) -> FrameworkEventProjection:
    """把尚未消费的框架事件投影为新的 observation/trace 列表。"""
    events = list(graph_state.get("governance_events", []))
    applied = max(0, int(graph_state.get("framework_events_applied", 0) or 0))
    observations = [
        dict(item)
        for item in graph_state.get("observations", [])
        if isinstance(item, dict)
    ]
    tool_trace = [
        dict(item)
        for item in graph_state.get("tool_trace", [])
        if isinstance(item, dict)
    ]
    model_index = sum(item.get("kind") == "model" for item in events[:applied])
    for event in events[applied:]:
        if event.get("kind") == "model":
            model_index += 1
            tool_trace.append(_model_trace(event, model_index))
            continue
        if event.get("counted_call"):
            observation, trace = _project_tool_event(
                event,
                step_id=len(observations) + 1,
            )
            observations.append(observation)
        else:
            trace = _guardrail_trace(event)
        tool_trace.append(trace)
    return {
        "observations": observations,
        "tool_trace": tool_trace,
        "framework_events_applied": len(events),
    }


def framework_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """从 append-only 事件计算兼容计数和连续指纹。"""
    calls = [
        item
        for item in events
        if item.get("kind") == "tool"
        and item.get("error_code") != "agent_batch_aborted"
    ]
    tools = [item for item in calls if item.get("counted_call")]
    errors = [item for item in tools if not item.get("ok")]

    def tail(key: str) -> tuple[str, int]:
        """计算指定指纹在调用事件尾部的连续次数。"""
        value = str(calls[-1].get(key) or "") if calls else ""
        count = 0
        for item in reversed(calls):
            if not value or str(item.get(key) or "") != value:
                break
            count += 1
        return value, count

    last_call, call_count = tail("call_fingerprint")
    last_error = str(errors[-1].get("error_fingerprint") or "") if errors else ""
    error_count = 0
    for item in reversed(tools):
        if item.get("ok") or str(item.get("error_fingerprint") or "") != last_error:
            break
        error_count += 1
    return {
        "iteration_count": sum(item.get("kind") == "model" for item in events),
        "tool_call_count": len(tools),
        "tool_error_count": len(errors),
        "last_tool_call_fingerprint": last_call,
        "consecutive_same_tool_call_count": call_count,
        "last_tool_error_fingerprint": last_error,
        "consecutive_same_tool_error_count": error_count,
    }


__all__ = [
    "FrameworkEventProjection",
    "build_framework_agent",
    "framework_metrics",
    "project_framework_events",
]
