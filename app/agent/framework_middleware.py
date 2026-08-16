"""动态工具与项目特有 Agent 治理 Middleware。"""

import json
import threading
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import ToolMessage

from app.agent.framework_tools import (
    find_last_ai_message,
    governance_block,
    parse_tool_result,
    resolve_tool_message,
    tool_call_fingerprint,
)
from app.agent.guardrails import (
    AgentStopReason,
    build_tool_governance_event,
    build_unknown_tool_call_fingerprint,
    count_consecutive_events,
    sum_governance_event_costs,
)
from app.config import AppSettings, get_settings


def _now_ms() -> int:
    """返回可序列化的 Unix 毫秒时间。"""
    return time.time_ns() // 1_000_000


_MODEL_GOVERNANCE_KEY = "_project_model_governance"
_TOOL_GOVERNANCE_KEY = "_project_tool_governance"


class ProjectToolGovernanceMiddleware(AgentMiddleware):
    """保留错误分类、循环检测、耗时预算和批次 fail-closed。"""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """初始化并发安全的同批事件协调器。"""
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._batch_events: dict[tuple[str, tuple[str, ...]], list[dict]] = {}

    def after_model(self, state, runtime):
        """消费一次模型私有标记并写入 append-only 治理事件。"""
        del runtime
        ai_message = find_last_ai_message(state.get("messages", []))
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
        elif not ai_message.tool_calls:
            updates["stop_reason"] = AgentStopReason.COMPLETED.value
        return updates

    async def aafter_model(self, state, runtime):
        """异步图复用同一模型治理事件消费逻辑。"""
        return self.after_model(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """在下一轮模型调用前执行停止和 RAG 交接检查。"""
        del runtime
        marker_events: list[dict[str, Any]] = []
        cleaned_messages: list[ToolMessage] = []
        knowledge_result: dict[str, Any] | None = None
        for item in state.get("messages", []):
            if not isinstance(item, ToolMessage) or not isinstance(
                item.artifact, dict
            ):
                continue
            marker = item.artifact.get(_TOOL_GOVERNANCE_KEY)
            if not isinstance(marker, dict):
                continue
            event = marker.get("event")
            if isinstance(event, dict):
                marker_events.append(dict(event))
            candidate = marker.get("knowledge_base_result")
            if isinstance(candidate, dict):
                knowledge_result = candidate
            cleaned_messages.append(
                item.model_copy(
                    update={
                        "artifact": None,
                        "status": (
                            "success"
                            if isinstance(event, dict) and event.get("ok") is True
                            else "error"
                        ),
                    }
                )
            )

        events = [*state.get("governance_events", []), *marker_events]
        reason = str(state.get("stop_reason") or "")
        if not reason:
            reason = next(
                (
                    str(item.get("stop_reason") or "")
                    for item in reversed(marker_events)
                    if item.get("stop_reason")
                ),
                "",
            )
        last_ai = find_last_ai_message(state.get("messages", []))
        trailing: list[object] = []
        if not reason and last_ai is not None:
            messages = state.get("messages", [])
            trailing = messages[messages.index(last_ai) + 1 :]
            if any(
                isinstance(item, ToolMessage)
                and str(item.content).startswith("Tool call limit exceeded")
                for item in trailing
            ):
                reason = AgentStopReason.MAX_TOOL_CALLS.value
        if last_ai is not None:
            latest_model_event = next(
                (
                    item
                    for item in reversed(events)
                    if item.get("kind") == "model"
                ),
                {},
            )
            available_tool_names = set(
                latest_model_event.get("available_tool_names", [])
            )
            calls_by_id = {
                str(call.get("id") or ""): call
                for call in last_ai.tool_calls
            }
            for item in trailing:
                if (
                    not isinstance(item, ToolMessage)
                    or isinstance(item.artifact, dict)
                    or item.tool_call_id not in calls_by_id
                    or str(item.content).startswith("Tool call limit exceeded")
                ):
                    continue
                call = calls_by_id[item.tool_call_id]
                args = call.get("args") or {}
                known_tool = str(call.get("name") or "") in available_tool_names
                result = parse_tool_result(
                    item,
                    known=known_tool,
                    name=str(call.get("name") or ""),
                    args=args,
                )
                event = build_tool_governance_event(
                    tool_call=call,
                    result=result,
                    call_fingerprint=build_unknown_tool_call_fingerprint(
                        tool_name=str(call.get("name") or ""),
                        tool_args=args,
                    ),
                    prior_events=events,
                    cost_ms=0,
                    known_tool=known_tool,
                    settings=self._settings,
                )
                result.update(
                    {
                        "call_fingerprint": event["call_fingerprint"],
                        "failure_kind": event["failure_kind"],
                        "error_source": event["error_source"],
                        "handler_executed": event["handler_executed"],
                        "stop_reason": event["stop_reason"],
                        "model_tool_call_id": call.get("id") or "",
                    }
                )
                marker_events.append(event)
                events.append(event)
                cleaned_messages.append(
                    item.model_copy(
                        update={
                            "content": json.dumps(
                                result,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "status": "error",
                        }
                    )
                )
                if not reason and event.get("stop_reason"):
                    reason = str(event["stop_reason"])
        updates: dict[str, Any] = {
            "agent_elapsed_ms": sum_governance_event_costs(events)
        }
        if marker_events:
            updates["governance_events"] = marker_events
            updates["messages"] = cleaned_messages
        if knowledge_result is not None:
            updates["knowledge_base_result"] = knowledge_result
        if reason:
            updates.update({"stop_reason": reason, "jump_to": "end"})
        elif knowledge_result is not None or state.get("knowledge_base_result"):
            updates["jump_to"] = "end"
        return updates

    async def abefore_model(self, state, runtime):
        """异步图复用同一停止、预算与 RAG 交接判断。"""
        return self.before_model(state, runtime)

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

    def _prepare_tool_call(self, request: ToolCallRequest):
        """计算同步与异步工具调用共用的治理上下文和前置拦截。"""
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
        if (
            not reason
            and elapsed_ms >= self._settings.agent_max_total_cost_ms
        ):
            reason = AgentStopReason.MAX_TOTAL_COST.value
        if (
            not reason
            and repeated
            > self._settings.agent_max_consecutive_same_tool_calls
        ):
            reason = AgentStopReason.REPEATED_TOOL_CALL.value
        if reason:
            message, event = governance_block(
                request,
                reason,
                fingerprint,
                aborted=bool(terminal),
            )
            self._remember(key, event, len(key[1]))
            blocked = message.model_copy(
                update={
                    "artifact": {
                        _TOOL_GOVERNANCE_KEY: {"event": event}
                    }
                }
            )
            return args, key, prior, fingerprint, blocked

        return args, key, prior, fingerprint, None

    def _complete_tool_call(
        self,
        request: ToolCallRequest,
        output: object,
        *,
        args: dict[str, Any],
        key: tuple[str, tuple[str, ...]],
        prior: list[dict],
        fingerprint: str,
        started: int,
    ) -> ToolMessage:
        """把同步或异步工具输出归一化为带内部标记的 ToolMessage。"""
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
            settings=self._settings,
        )
        return self._finish_tool_call(request, message, result, event, key)

    def govern_tool_call(self, request: ToolCallRequest, handler):
        """在本地或 MCP 工具同步执行前后应用项目治理规则。"""
        args, key, prior, fingerprint, blocked = self._prepare_tool_call(request)
        if blocked is not None:
            return blocked

        started = _now_ms()
        output = handler(request)
        return self._complete_tool_call(
            request,
            output,
            args=args,
            key=key,
            prior=prior,
            fingerprint=fingerprint,
            started=started,
        )

    async def agovern_tool_call(self, request: ToolCallRequest, handler):
        """在异步工具执行前后应用与同步入口相同的治理规则。"""
        args, key, prior, fingerprint, blocked = self._prepare_tool_call(request)
        if blocked is not None:
            return blocked

        started = _now_ms()
        output = await handler(request)
        return self._complete_tool_call(
            request,
            output,
            args=args,
            key=key,
            prior=prior,
            fingerprint=fingerprint,
            started=started,
        )

    def _finish_tool_call(
        self, request, message, result, event, key
    ) -> ToolMessage:
        """补充兼容字段，并把治理事件放入内部 ToolMessage artifact。"""
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
        self._remember(key, event, len(key[1]))
        marker: dict[str, Any] = {"event": event}
        if request.tool_call["name"] == "search_knowledge_base" and event["ok"]:
            output_value = result.get("output")
            if (
                isinstance(output_value, dict)
                and isinstance(output_value.get("rag_result"), dict)
            ):
                marker["knowledge_base_result"] = result
        return message.model_copy(
            update={
                "content": json.dumps(
                    result, ensure_ascii=False, sort_keys=True, default=str
                ),
                "status": "success" if event["ok"] else "error",
                "artifact": {_TOOL_GOVERNANCE_KEY: marker},
            }
        )

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

    async def aafter_agent(self, state, runtime):
        """异步图复用同一最终停止原因与累计耗时归一化。"""
        return self.after_agent(state, runtime)
