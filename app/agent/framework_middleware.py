"""LangChain 原生工具生命周期与项目特有 Agent 治理。"""

import json
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest, hook_config
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.agent.guardrails import (
    AgentStopReason,
    build_tool_governance_event,
    build_unknown_tool_call_fingerprint,
    count_consecutive_events,
    sum_governance_event_costs,
)
from app.config import AppSettings, get_settings
from app.tools.errors import ToolErrorCode
from app.tools.execution import (
    enter_framework_execution,
    exit_framework_execution,
    framework_execution_started,
    invalid_arguments_result,
)
from app.tools.mcp.serialization import normalize_mcp_tool_message
from app.tools.policy import build_tool_args_fingerprint
from app.tools.result import tool_error


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


def parse_tool_result(
    message: ToolMessage,
    *,
    known: bool,
    name: str,
    args: dict,
) -> dict[str, Any]:
    """读取标准 ToolMessage；框架校验错误映射为项目稳定错误码。"""
    mcp_result = normalize_mcp_tool_message(message, tool_name=name, args=args)
    if mcp_result is not None:
        return mcp_result
    try:
        value = (
            json.loads(message.content) if isinstance(message.content, str) else None
        )
    except (TypeError, ValueError):
        value = None
    if isinstance(value, dict) and "ok" in value:
        return value
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
    """从原生 ToolMessage 或 Command 中取得本次配对消息。"""
    if isinstance(output, ToolMessage) and output.tool_call_id == call_id:
        return output
    if isinstance(output, Command) and isinstance(output.update, dict):
        messages = output.update.get("messages", [])
        matches = [
            item
            for item in messages
            if isinstance(item, ToolMessage) and item.tool_call_id == call_id
        ]
        if len(matches) == 1:
            return matches[0]
    raise TypeError("governed tool must return its paired ToolMessage")


def _replace_command_message(
    output: Command[Any], message: ToolMessage
) -> dict[str, Any]:
    """保留工具原有 Command 更新，并替换治理后的配对消息。"""
    if not isinstance(output.update, dict):
        raise TypeError("governed tool Command must use a dict state update")
    updates = dict(output.update)
    raw_messages = updates.get("messages", [])
    if not isinstance(raw_messages, (list, tuple)):
        raise TypeError("governed tool Command messages must be a sequence")
    replaced = False
    messages: list[object] = []
    for item in raw_messages:
        if isinstance(item, ToolMessage) and item.tool_call_id == message.tool_call_id:
            messages.append(message)
            replaced = True
        else:
            messages.append(item)
    if not replaced:
        raise TypeError("governed tool Command lost its paired ToolMessage")
    updates["messages"] = messages
    return updates


def _tool_state_command(
    output: ToolMessage | Command[Any],
    message: ToolMessage,
    event: dict[str, Any],
    *,
    knowledge_base_result: dict[str, Any] | None = None,
) -> Command[Any]:
    """把工具消息与项目治理结果直接写入 FrameworkAgentState。"""
    if isinstance(output, Command):
        updates = _replace_command_message(output, message)
        graph = output.graph
        resume = output.resume
        goto = output.goto
    else:
        updates = {"messages": [message]}
        graph = None
        resume = None
        goto = ()

    existing_events = updates.get("governance_events", [])
    if not isinstance(existing_events, list):
        raise TypeError("governance_events update must be a list")
    updates["governance_events"] = [*existing_events, event]
    if event.get("stop_reason"):
        updates["stop_reason"] = str(event["stop_reason"])
    if knowledge_base_result is not None:
        updates["knowledge_base_result"] = knowledge_base_result
    return Command(graph=graph, update=updates, resume=resume, goto=goto)


class ProjectToolExecutionMiddleware(AgentMiddleware):
    """仅为本地 ProjectStructuredTool 建立 retry scope 和校验错误契约。"""

    @staticmethod
    def _project_tool(request: ToolCallRequest):
        from app.agent.framework_tools import ProjectStructuredTool

        return (
            request.tool
            if isinstance(request.tool, ProjectStructuredTool)
            else None
        )

    @staticmethod
    def _validated_output(request, tool, output):
        if framework_execution_started() or not isinstance(output, ToolMessage):
            return output
        context = request.runtime.context
        if not isinstance(context, type(tool.project_context)):
            context = tool.project_context
        raw_args = request.tool_call.get("args") or {}
        result = invalid_arguments_result(
            spec=tool.project_spec,
            tool_args=raw_args if isinstance(raw_args, dict) else {},
            context=context,
            tool_call_id=str(request.tool_call.get("id") or ""),
            settings=tool.project_settings,
            audit_sink=tool.project_audit_sink,
        )
        return output.model_copy(
            update={
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "status": "error",
            }
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """本地工具使用 ContextVar 跨 framework retry 共享 attempt 状态。"""
        tool = self._project_tool(request)
        if tool is None:
            return handler(request)
        token = enter_framework_execution()
        try:
            return self._validated_output(request, tool, handler(request))
        finally:
            exit_framework_execution(token)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        """异步 Agent 复用相同本地执行边界，MCP BaseTool 直接跳过。"""
        tool = self._project_tool(request)
        if tool is None:
            return await handler(request)
        token = enter_framework_execution()
        try:
            output = await handler(request)
            return self._validated_output(request, tool, output)
        finally:
            exit_framework_execution(token)


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
        elif not ai_message.tool_calls:
            updates["stop_reason"] = AgentStopReason.COMPLETED.value
        return updates

    async def aafter_model(self, state, runtime):
        """异步图复用同一模型治理事件消费逻辑。"""
        return self.after_model(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        """依据 State 中的停止原因和 RAG 结果决定下一步。"""
        del runtime
        events = list(state.get("governance_events", []))
        reason = str(state.get("stop_reason") or "")
        if not reason and _has_tool_call_limit_message(state.get("messages", [])):
            reason = AgentStopReason.MAX_TOOL_CALLS.value
        updates: dict[str, Any] = {
            "agent_elapsed_ms": sum_governance_event_costs(events)
        }
        if reason:
            updates.update({"stop_reason": reason, "jump_to": "end"})
        elif state.get("knowledge_base_result"):
            updates["jump_to"] = "end"
        return updates

    async def abefore_model(self, state, runtime):
        """异步图复用同一停止、预算与 RAG 交接判断。"""
        return self.before_model(state, runtime)

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
        """计算同步与异步工具调用共用的治理上下文和前置拦截。"""
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
        elapsed_ms = sum_governance_event_costs(prior)
        if not reason and elapsed_ms >= self._settings.agent_max_total_cost_ms:
            reason = AgentStopReason.MAX_TOTAL_COST.value
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
        output: ToolMessage | Command[Any],
        *,
        args: dict[str, Any],
        key: tuple[str, tuple[str, ...]],
        prior: list[dict],
        fingerprint: str,
        started: int,
    ) -> Command[Any]:
        """把工具输出转换为直接 State Update 的治理 Command。"""
        message = _resolve_tool_message(output, request.tool_call["id"])
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
        return self._finish_tool_call(request, output, message, result, event, key)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """通过 LangChain 原生同步 Hook 应用项目 Agent 治理。"""
        args, key, prior, fingerprint, blocked = self._prepare_tool_call(request)
        if blocked is not None:
            message, event = blocked
            return self._finish_tool_call(
                request,
                message,
                message,
                event["result"],
                event,
                key,
                remember=False,
            )

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
                request,
                message,
                message,
                event["result"],
                event,
                key,
                remember=False,
            )

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
        self,
        request: ToolCallRequest,
        output: ToolMessage | Command[Any],
        message: ToolMessage,
        result: dict[str, Any],
        event: dict[str, Any],
        key: tuple[str, tuple[str, ...]],
        *,
        remember: bool = True,
    ) -> Command[Any]:
        """补充兼容字段，并生成原生 Command State Update。"""
        result.update(
            {
                "call_fingerprint": event.get("call_fingerprint", ""),
                "failure_kind": event.get("failure_kind", ""),
                "error_source": event.get(
                    "error_source", result.get("error_source", "")
                ),
                "handler_executed": event.get(
                    "handler_executed", result.get("handler_executed", False)
                ),
                "stop_reason": event.get("stop_reason", ""),
                "model_tool_call_id": request.tool_call["id"],
            }
        )
        if remember:
            self._remember(key, event, len(key[1]))
        governed_message = message.model_copy(
            update={
                "content": json.dumps(
                    result, ensure_ascii=False, sort_keys=True, default=str
                ),
                "status": "success" if event.get("ok") is True else "error",
            }
        )
        knowledge_result = None
        if request.tool_call["name"] == "search_knowledge_base" and event.get("ok"):
            output_value = result.get("output")
            if (
                isinstance(output_value, dict)
                and isinstance(output_value.get("rag_result"), dict)
            ):
                knowledge_result = result
        return _tool_state_command(
            output,
            governed_message,
            event,
            knowledge_base_result=knowledge_result,
        )

    def after_agent(self, state, runtime):
        """归一化框架结束原因并保存最终累计耗时。"""
        del runtime
        events = list(state.get("governance_events", []))
        reason = str(state.get("stop_reason") or "")
        last_ai = _find_last_ai_message(state.get("messages", []))
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
