"""直接执行 ``create_agent.astream`` 并投影安全流式事件。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agent.final_response import FinalAgentResponse
from app.agent.results import (
    build_agent_runtime_error_result,
    build_framework_agent_result,
    build_short_circuit_result,
)
from app.agent.runtime import (
    prepare_agent_runtime_invocation,
)
from app.common.logging import get_logger
from app.contracts.chat import AgentResultStatus
from app.tools.context import ToolInvocationContext


logger = get_logger("chat_service")

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_STREAM_MODES = ("updates", "messages", "values")


@dataclass(frozen=True)
class AgentRuntimeStreamEvent:
    """Agent Runtime 向 ChatService 暴露的最小流式事件。"""

    event: str
    data: dict[str, Any]
    key: str = ""


def _safe_tool_name(value: object) -> str:
    """只接受 Registry 风格的工具名，拒绝路径和自由文本。"""
    name = str(value or "").strip()
    return name if _TOOL_NAME_PATTERN.fullmatch(name) else ""


def _safe_non_negative_int(value: object) -> int:
    """把内部耗时安全规整为非负整数。"""
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _namespace_is_agent(namespace: tuple[str, ...]) -> bool:
    """接受根 ``create_agent``，同时显式拒绝任何 RAG 子图 namespace。"""
    if any(part.split(":", 1)[0] == "rag_subgraph" for part in namespace):
        return False
    return not namespace or any(
        part.split(":", 1)[0] == "framework_agent"
        for part in namespace
    )


def _tool_calls_from_message(message: AIMessage) -> list[Mapping[str, Any]]:
    """读取工具名和调用 ID，不读取或复制任何工具参数。"""
    calls: list[Mapping[str, Any]] = []
    for value in (
        getattr(message, "tool_calls", None),
        getattr(message, "tool_call_chunks", None),
    ):
        if not isinstance(value, Sequence):
            continue
        calls.extend(item for item in value if isinstance(item, Mapping))
    return calls


def _project_message_part(
    data: object,
    namespace: tuple[str, ...],
) -> list[AgentRuntimeStreamEvent]:
    """按消息类型、block、namespace 和 metadata 白名单投影模型消息。"""
    if not _namespace_is_agent(namespace):
        return []
    if not isinstance(data, tuple) or len(data) != 2:
        return []
    message, metadata = data
    if isinstance(message, ToolMessage) or not isinstance(
        message,
        (AIMessage, AIMessageChunk),
    ):
        return []
    if not isinstance(metadata, Mapping):
        return []
    if metadata.get("langgraph_node") != "model":
        return []
    if metadata.get("lc_agent_name") not in (None, "", "framework_agent"):
        return []

    message_id = str(
        getattr(message, "id", "")
        or metadata.get("run_id")
        or metadata.get("langgraph_checkpoint_ns")
        or ""
    )
    calls = _tool_calls_from_message(message)
    projected: list[AgentRuntimeStreamEvent] = []
    for call in calls:
        if call.get("name") == FinalAgentResponse.__name__:
            continue
        tool_name = _safe_tool_name(call.get("name"))
        if not tool_name:
            continue
        call_id = str(call.get("id") or message_id or tool_name)
        projected.append(
            AgentRuntimeStreamEvent(
                event="tool",
                data={
                    "tool_name": tool_name,
                    "phase": "started",
                    "status": "",
                },
                key=f"tool:{call_id}:started",
            )
        )

    # 普通模型文本和 schema args 均不是权威答案；待最终 grounding 校验后发送。
    return projected


def _project_tool_completions(
    update: Mapping[str, Any],
) -> list[AgentRuntimeStreamEvent]:
    """从治理事件中只投影工具名、状态和耗时。"""
    events = update.get("governance_events")
    if not isinstance(events, list):
        return []
    projected: list[AgentRuntimeStreamEvent] = []
    for item in events:
        if not isinstance(item, Mapping) or item.get("kind") != "tool":
            continue
        tool_name = _safe_tool_name(item.get("name"))
        if not tool_name:
            continue
        call_id = str(
            item.get("tool_call_id")
            or item.get("call_fingerprint")
            or tool_name
        )
        projected.append(
            AgentRuntimeStreamEvent(
                event="tool",
                data={
                    "tool_name": tool_name,
                    "phase": "completed",
                    "status": "success" if item.get("ok") is True else "error",
                    "cost_ms": _safe_non_negative_int(item.get("cost_ms")),
                },
                key=f"tool:{call_id}:completed",
            )
        )
    return projected


def _project_nested_update(value: object) -> list[AgentRuntimeStreamEvent]:
    """递归寻找原生 Command 更新，到达治理事件字段后停止深入。"""
    if isinstance(value, Mapping):
        if "governance_events" in value:
            return _project_tool_completions(value)
        projected: list[AgentRuntimeStreamEvent] = []
        for nested in value.values():
            projected.extend(_project_nested_update(nested))
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        projected: list[AgentRuntimeStreamEvent] = []
        for nested in value:
            projected.extend(_project_nested_update(nested))
        return projected
    return []


def _project_update_part(data: object) -> list[AgentRuntimeStreamEvent]:
    """从 ``create_agent`` updates 中投影工具完成状态。"""
    if not isinstance(data, Mapping):
        return []
    projected: list[AgentRuntimeStreamEvent] = []
    for raw_update in data.values():
        projected.extend(_project_nested_update(raw_update))
    return projected


def project_agent_stream_part(part: object) -> list[AgentRuntimeStreamEvent]:
    """适配 LangGraph v2 part，拒绝未知模式和未知结构。"""
    if not isinstance(part, Mapping):
        return []
    stream_type = part.get("type")
    namespace_value = part.get("ns", ())
    namespace = (
        tuple(str(item) for item in namespace_value)
        if isinstance(namespace_value, (list, tuple))
        else ()
    )
    if stream_type == "messages":
        return _project_message_part(part.get("data"), namespace)
    if stream_type == "updates":
        return _project_update_part(part.get("data"))
    return []


async def _close_async_iterator(iterator: object) -> None:
    """尽最大努力关闭 Agent 流，使断开和取消向下传播。"""
    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()


async def stream_agent(
    *,
    write_bad_cases: Callable[[dict[str, Any]], None] | None,
    top_k: int,
    decompose_top_k: int,
    trace_id: str,
    session_id: str | None,
    bad_case: dict[str, Any] | None,
    original_question: str,
    effective_question: str,
    tool_context: ToolInvocationContext | None = None,
) -> AsyncIterator[AgentRuntimeStreamEvent]:
    """路由后直接流式执行 ``create_agent`` 并产出安全事件。"""
    resolved_original = (original_question or "").strip()
    resolved_effective = (effective_question or "").strip()
    agent_stream: object | None = None
    agent_stream_completed = False
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
        yield AgentRuntimeStreamEvent(
            event="progress",
            data={"stage": "routing", "status": "completed"},
            key="progress:routing:completed",
        )
        if invocation.agent is None:
            result = build_short_circuit_result(invocation)
            yield AgentRuntimeStreamEvent(
                event="result",
                data={"result": result},
                key="result:final",
            )
            return

        agent_stream = invocation.agent.astream(
            invocation.agent_input,
            config=invocation.agent_config,
            context=invocation.tool_context,
            stream_mode=list(_STREAM_MODES),
            subgraphs=False,
            version="v2",
        )
        final_state: Mapping[str, Any] | None = None
        seen_event_keys: set[str] = set()

        async for part in agent_stream:
            if isinstance(part, Mapping) and part.get("type") == "values":
                value = part.get("data")
                if isinstance(value, Mapping):
                    final_state = value
                continue
            for event in project_agent_stream_part(part):
                if event.key and event.key in seen_event_keys:
                    continue
                if event.key:
                    seen_event_keys.add(event.key)
                yield event
                if (
                    event.event == "tool"
                    and event.data.get("tool_name") == "search_knowledge_base"
                ):
                    phase = str(event.data.get("phase") or "")
                    if phase in {"started", "completed"}:
                        progress = AgentRuntimeStreamEvent(
                            event="progress",
                            data={"stage": "rag", "status": phase},
                            key=f"progress:rag:{phase}",
                        )
                        if progress.key not in seen_event_keys:
                            seen_event_keys.add(progress.key)
                            yield progress

        agent_stream_completed = True
        if final_state is None:
            raise RuntimeError("create_agent stream did not produce final state")
        result = build_framework_agent_result(invocation, final_state)
        if result["status"] in {AgentResultStatus.OK, AgentResultStatus.NO_RELEVANT_INFO}:
            yield AgentRuntimeStreamEvent(
                event="answer_delta",
                data={"text": result["answer"]},
                key="answer:final",
            )
        logger.info(
            "[trace_id=%s] [agent] runtime_end status=%s streaming=true",
            trace_id,
            result.get("status", ""),
        )
        yield AgentRuntimeStreamEvent(
            event="result",
            data={"result": result},
            key="result:final",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "[trace_id=%s] [agent] runtime_failed streaming=true",
            trace_id,
        )
        result = build_agent_runtime_error_result(
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original,
            effective_question=resolved_effective,
            exc=exc,
        )
        yield AgentRuntimeStreamEvent(
            event="result",
            data={"result": result},
            key="result:error",
        )
    finally:
        if agent_stream is not None and not agent_stream_completed:
            await _close_async_iterator(agent_stream)


__all__ = [
    "AgentRuntimeStreamEvent",
    "project_agent_stream_part",
    "stream_agent",
]
