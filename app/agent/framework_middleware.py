"""动态工具与项目特有 Agent 治理 Middleware。"""

import json
import threading
import time
from dataclasses import replace
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent.framework_tools import (
    find_last_ai_message,
    governance_block,
    parse_tool_result,
    resolve_tool_message,
    tool_call_fingerprint,
)
from app.agent.guardrails import AgentStopReason
from app.agent.runtime import (
    build_tool_governance_event,
    count_consecutive_events,
    sum_governance_event_costs,
)
from app.config import Settings


def _now_ms() -> int:
    """返回可序列化的 Unix 毫秒时间。"""
    return time.time_ns() // 1_000_000


class ProjectToolGovernanceMiddleware(AgentMiddleware):
    """保留错误分类、循环检测、耗时预算和批次 fail-closed。"""

    def __init__(self) -> None:
        """初始化并发安全的同批事件协调器。"""
        self._lock = threading.Lock()
        self._batch_events: dict[tuple[str, tuple[str, ...]], list[dict]] = {}

    def wrap_model_call(self, request: ModelRequest, handler):
        """调用模型并记录结果、耗时和可见工具。"""
        started = _now_ms()
        try:
            response = handler(request)
        except Exception as exc:
            event = {
                "kind": "model",
                "ok": False,
                "cost_ms": max(0, _now_ms() - started),
                "error_type": type(exc).__name__,
            }
            return ExtendedModelResponse(
                ModelResponse(result=[AIMessage(content="Agent model unavailable")]),
                Command(
                    update={
                        "governance_events": [event],
                        "stop_reason": "agent_model_call_failed",
                    }
                ),
            )
        ai_message = find_last_ai_message(response.result)
        event = {
            "kind": "model",
            "ok": True,
            "cost_ms": max(0, _now_ms() - started),
            "has_tool_calls": bool(ai_message and ai_message.tool_calls),
            "requested_tool_call_count": len(ai_message.tool_calls)
            if ai_message
            else 0,
            "available_tool_names": [tool.name for tool in request.tools],
        }
        return ExtendedModelResponse(
            response,
            Command(update={"governance_events": [event]}),
        )

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """在下一轮模型调用前执行停止和 RAG 交接检查。"""
        events = list(state.get("governance_events", []))
        reason = str(state.get("stop_reason") or "")
        last_ai = find_last_ai_message(state.get("messages", []))
        if not reason and last_ai is not None:
            messages = state.get("messages", [])
            trailing = messages[messages.index(last_ai) + 1 :]
            if any(
                isinstance(item, ToolMessage)
                and str(item.content).startswith("Tool call limit exceeded")
                for item in trailing
            ):
                reason = AgentStopReason.MAX_TOOL_CALLS.value
        updates: dict[str, Any] = {
            "agent_elapsed_ms": sum_governance_event_costs(events)
        }
        if reason:
            updates.update({"stop_reason": reason, "jump_to": "end"})
        elif state.get("pending_rag_result"):
            updates["jump_to"] = "end"
        return updates

    def _batch_key(self, request: ToolCallRequest) -> tuple[str, tuple[str, ...]]:
        """使用 trace 和模型调用 ID 标识同一批工具调用。"""
        ai_message = find_last_ai_message(request.state.get("messages", []))
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

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """在框架工具执行前后应用项目特有治理规则。"""
        args = request.tool_call.get("args") or {}
        key = self._batch_key(request)
        with self._lock:
            batch_events = list(self._batch_events.get(key, []))
        prior = [*request.state.get("governance_events", []), *batch_events]
        fingerprint = tool_call_fingerprint(request, args)
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
        elapsed_ms = sum_governance_event_costs(prior)
        if not reason and elapsed_ms >= Settings.AGENT_MAX_TOTAL_COST_MS:
            reason = AgentStopReason.MAX_TOTAL_COST.value
        if not reason and repeated > Settings.AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS:
            reason = AgentStopReason.REPEATED_TOOL_CALL.value
        if reason:
            message, event = governance_block(
                request,
                reason,
                fingerprint,
                aborted=bool(terminal),
            )
            self._remember(key, event, len(key[1]))
            return Command(
                update={
                    "messages": [message],
                    "governance_events": [event],
                    "stop_reason": reason,
                }
            )

        started = _now_ms()
        output = handler(request)
        message = resolve_tool_message(output, request.tool_call["id"])
        result = parse_tool_result(
            message,
            known=request.tool is not None,
            name=request.tool_call["name"],
            args=args,
        )
        event = build_tool_governance_event(
            tool_call=request.tool_call,
            result=result,
            call_fingerprint=fingerprint,
            prior_events=prior,
            cost_ms=_now_ms() - started,
            known_tool=request.tool is not None,
        )
        return self._finish_tool_call(request, output, message, result, event, key)

    def _finish_tool_call(
        self, request, output, message, result, event, key
    ) -> Command:
        """补充兼容结果字段，并合并框架 ToolMessage 或 Command。"""
        result.update(
            {
                "call_fingerprint": event["call_fingerprint"],
                "failure_kind": event["failure_kind"],
                "error_source": event["error_source"],
                "handler_executed": event["handler_executed"],
                "stop_reason": event["stop_reason"],
                "model_tool_call_id": request.tool_call["id"],
            }
        )
        message = message.model_copy(
            update={
                "content": json.dumps(
                    result, ensure_ascii=False, sort_keys=True, default=str
                ),
                "status": "success" if event["ok"] else "error",
            }
        )
        self._remember(key, event, len(key[1]))
        updates = {"governance_events": [event]}
        if event.get("stop_reason"):
            updates["stop_reason"] = event["stop_reason"]
        if request.tool_call["name"] == "search_knowledge_base" and event["ok"]:
            updates["pending_rag_result"] = result
        if isinstance(output, Command):
            command_updates = dict(output.update or {})
            command_updates.update(updates)
            command_updates["messages"] = [
                item if not isinstance(item, ToolMessage) else message
                for item in command_updates.get("messages", [])
            ]
            return replace(output, update=command_updates)
        updates["messages"] = [message]
        return Command(update=updates)

    def after_agent(self, state, runtime):
        """归一化框架结束原因并保存最终累计耗时。"""
        del runtime
        events = list(state.get("governance_events", []))
        reason = str(state.get("stop_reason") or "")
        last_ai = find_last_ai_message(state.get("messages", []))
        if (
            not reason
            and last_ai
            and str(last_ai.content).startswith("Model call limits exceeded")
        ):
            reason = AgentStopReason.MAX_ITERATIONS.value
        return {
            "agent_elapsed_ms": sum_governance_event_costs(events),
            "stop_reason": reason,
        }
