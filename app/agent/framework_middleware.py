"""LangChain 原生工具生命周期与项目特有 Agent 治理。"""

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.types import ToolCallRequest, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent.final_response import FinalAgentResponse
from app.agent.guardrails import (
    AgentStopReason,
    build_tool_governance_event,
    build_unknown_tool_call_fingerprint,
    count_consecutive_events,
)
from app.config import AppSettings, get_settings
from app.tools.errors import ToolErrorCode
from app.tools.policy import build_tool_args_fingerprint
from app.tools.result import tool_content, tool_error


def _now_ms() -> int:
    """返回可序列化的 Unix 毫秒时间。"""
    return time.time_ns() // 1_000_000


_MODEL_GOVERNANCE_KEY = "_project_model_governance"
_TOOL_CALL_LIMIT_PREFIX = "Tool call limit exceeded"


def _find_last_ai_message(messages: list[object]) -> AIMessage | None:
    """返回消息历史中最近的 AIMessage。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _has_tool_call_limit_message(messages: list[object]) -> bool:
    """只检查最近模型调用后的 LangChain 工具上限消息。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return False
        if isinstance(message, ToolMessage) and str(message.content).startswith(
            _TOOL_CALL_LIMIT_PREFIX
        ):
            return True
    return False


class ProjectToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """1.3.14 原生限流会计入合成响应；仅过滤其计数输入，复用原生限流。"""

    def after_model(self, state, runtime):
        messages = list(state.get("messages", []))
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage):
                messages[index] = message.model_copy(update={"tool_calls": [
                    call for call in message.tool_calls
                    if call["name"] != FinalAgentResponse.__name__
                ]})
                break
        update = super().after_model({**state, "messages": messages}, runtime)
        if update and update.get("messages"):
            update["stop_reason"] = AgentStopReason.MAX_TOOL_CALLS.value
        return update


def parse_tool_result(
    message: ToolMessage,
    *,
    known: bool,
    name: str,
    args: dict,
) -> dict[str, Any]:
    """读取标准 ToolMessage；框架校验错误映射为项目稳定错误码。"""
    if isinstance(message.artifact, dict) and "ok" in message.artifact:
        return message.artifact
    code = ToolErrorCode.INVALID_ARGUMENTS if known else ToolErrorCode.UNKNOWN_TOOL
    return tool_error(name, args, str(message.content), error_code=code.value)


def _tool_call_fingerprint(request: ToolCallRequest, args: dict) -> str:
    """注册工具复用 Policy 指纹；未知工具仅处理模型 JSON。"""
    metadata = request.tool.metadata if request.tool is not None else None
    version = str((metadata or {}).get("tool_version") or "")
    if version:
        try:
            return build_tool_args_fingerprint(
                tool_name=request.tool_call["name"],
                tool_version=version,
                args=args,
            )
        except (TypeError, ValueError):
            pass
    return build_unknown_tool_call_fingerprint(
        tool_name=request.tool_call["name"], tool_args=args
    )


def _governance_block(
    request: ToolCallRequest,
    reason: str,
    fingerprint: str,
    *,
    aborted: bool,
) -> tuple[ToolMessage, dict[str, Any]]:
    """为未执行的同批调用补齐唯一的配对 ToolMessage。"""
    code = "agent_batch_aborted" if aborted else reason
    args = request.tool_call.get("args") or {}
    result = tool_error(
        request.tool_call["name"],
        args,
        f"Agent governance stopped: {reason}",
        error_code=code,
    )
    result.update(
        {
            "handler_executed": False,
            "stop_reason": reason,
            "error_source": "agent_guardrail",
        }
    )
    message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False, sort_keys=True),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )
    event = {
        "kind": "tool",
        "name": request.tool_call["name"],
        "args": args,
        "result": result,
        "ok": False,
        "counted_call": False,
        "call_fingerprint": fingerprint,
        "error_code": code,
        "stop_reason": reason,
        "cost_ms": 0,
        "tool_call_id": str(request.tool_call.get("id") or ""),
    }
    return message, event


def _resolve_tool_message(output: object, call_id: str) -> ToolMessage:
    """项目工具只返回配对 ToolMessage；Command 由治理层生成。"""
    if isinstance(output, ToolMessage) and output.tool_call_id == call_id:
        return output
    raise TypeError("governed tool must return its paired ToolMessage")


def _tool_state_command(
    message: ToolMessage,
    event: dict[str, Any],
) -> Command[Any]:
    """把工具消息与项目治理结果直接写入 FrameworkAgentState。"""
    updates: dict[str, Any] = {"messages": [message], "governance_events": [event]}
    if event.get("stop_reason"):
        updates["stop_reason"] = str(event["stop_reason"])
    return Command(update=updates)


class ProjectToolGovernanceMiddleware(AgentMiddleware):
    """保留错误分类、循环检测、批次 fail-closed。"""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """初始化并发安全的同批事件协调器。"""
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._batch_events: dict[tuple[str, tuple[str, ...]], list[dict]] = {}

    async def aafter_model(self, state, runtime):
        """消费一次模型私有标记并写入 append-only 治理事件。"""
        del runtime
        ai_message = _find_last_ai_message(state.get("messages", []))
        if ai_message is None:
            return None
        raw_event = ai_message.additional_kwargs.get(_MODEL_GOVERNANCE_KEY)
        if not isinstance(raw_event, dict):
            return None
        additional_kwargs = dict(ai_message.additional_kwargs)
        additional_kwargs.pop(_MODEL_GOVERNANCE_KEY, None)
        updates: dict[str, Any] = {
            "messages": [
                ai_message.model_copy(
                    update={"additional_kwargs": additional_kwargs}
                )
            ],
            "governance_events": [dict(raw_event)],
        }
        if raw_event.get("ok") is False:
            updates["stop_reason"] = "agent_model_call_failed"
        elif not ai_message.tool_calls or (
            state.get("structured_response") is not None
            and not raw_event.get("has_tool_calls")
        ):
            updates["stop_reason"] = AgentStopReason.COMPLETED.value
        return updates

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state, runtime):
        """依据 State 中的治理停止原因决定是否结束。"""
        del runtime
        reason = str(state.get("stop_reason") or "")
        if not reason and _has_tool_call_limit_message(state.get("messages", [])):
            reason = AgentStopReason.MAX_TOOL_CALLS.value
        updates: dict[str, Any] = {}
        if reason:
            updates.update({"stop_reason": reason, "jump_to": "end"})
        return updates

    def _batch_key(self, request: ToolCallRequest) -> tuple[str, tuple[str, ...]]:
        """使用 trace 和模型调用 ID 标识同一批工具调用。"""
        ai_message = _find_last_ai_message(request.state.get("messages", []))
        tool_calls = ai_message.tool_calls if ai_message else []
        call_ids = tuple(call["id"] for call in tool_calls)
        return request.runtime.context.trace_id, call_ids

    def _remember(self, key, event, batch_size: int) -> None:
        """记录批次事件，并在整批完成后释放临时状态。"""
        with self._lock:
            events = self._batch_events.setdefault(key, [])
            events.append(event)
            if len(events) >= batch_size:
                self._batch_events.pop(key, None)

    def _prepare_tool_call(self, request: ToolCallRequest):
        """计算工具调用的治理上下文和前置拦截。"""
        args = request.tool_call.get("args") or {}
        key = self._batch_key(request)
        with self._lock:
            batch_events = list(self._batch_events.get(key, []))
        prior = [*request.state.get("governance_events", []), *batch_events]
        fingerprint = _tool_call_fingerprint(request, args)
        terminal = next(
            (
                item.get("stop_reason")
                for item in reversed(batch_events)
                if item.get("stop_reason")
            ),
            "",
        )
        repeated = (
            count_consecutive_events(
                prior,
                key="call_fingerprint",
                value=fingerprint,
            )
            + 1
        )
        reason = str(terminal or "")
        if (
            not reason
            and repeated > self._settings.agent_max_consecutive_same_tool_calls
        ):
            reason = AgentStopReason.REPEATED_TOOL_CALL.value
        blocked = None
        if reason:
            blocked = _governance_block(
                request,
                reason,
                fingerprint,
                aborted=bool(terminal),
            )
            self._remember(key, blocked[1], len(key[1]))
        return args, key, prior, fingerprint, blocked

    def _complete_tool_call(
        self,
        request: ToolCallRequest,
        message: ToolMessage,
        *,
        args: dict[str, Any],
        key: tuple[str, tuple[str, ...]],
        prior: list[dict],
        fingerprint: str,
        started: int,
    ) -> Command[Any]:
        """把工具输出转换为直接 State Update 的治理 Command。"""
        result = parse_tool_result(
            message,
            known=request.tool is not None,
            name=request.tool_call["name"],
            args=args,
        )
        if message.artifact is not result:
            message = message.model_copy(update={"content": tool_content(result), "artifact": result})
        event = build_tool_governance_event(
            tool_call=request.tool_call,
            result=result,
            call_fingerprint=fingerprint,
            prior_events=prior,
            cost_ms=_now_ms() - started,
            known_tool=request.tool is not None,
            settings=self._settings,
        )
        return self._finish_tool_call(message, event, key)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        """通过 LangChain 原生异步 Hook 应用相同治理。"""
        args, key, prior, fingerprint, blocked = self._prepare_tool_call(request)
        if blocked is not None:
            message, event = blocked
            return self._finish_tool_call(
                message,
                event,
                key,
                remember=False,
            )

        started = _now_ms()
        try:
            output = await handler(request)
        except asyncio.CancelledError:
            with self._lock:
                self._batch_events.pop(key, None)
            raise
        return self._complete_tool_call(
            request,
            _resolve_tool_message(output, request.tool_call["id"]),
            args=args,
            key=key,
            prior=prior,
            fingerprint=fingerprint,
            started=started,
        )

    def _finish_tool_call(
        self,
        message: ToolMessage,
        event: dict[str, Any],
        key: tuple[str, tuple[str, ...]],
        *,
        remember: bool = True,
    ) -> Command[Any]:
        """内部治理只写 State，模型消息不重复解析或序列化。"""
        if remember:
            self._remember(key, event, len(key[1]))
        governed_message = message.model_copy(
            update={
                "status": "success" if event.get("ok") is True else "error",
            }
        )
        return _tool_state_command(governed_message, event)

    async def aafter_agent(self, state, runtime):
        """归一化框架结束原因。"""
        del runtime
        reason = str(state.get("stop_reason") or "")
        last_ai = _find_last_ai_message(state.get("messages", []))
        if (
            not reason
            and last_ai
            and str(last_ai.content).startswith("Model call limits exceeded")
        ):
            reason = AgentStopReason.MAX_ITERATIONS.value
        return {
            "stop_reason": reason,
        }
