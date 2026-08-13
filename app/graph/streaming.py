"""LangGraph 原始流的异步执行与安全事件投影。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agent.results import validate_agent_result_payload
from app.common.logging import get_logger
from app.graph.runtime import (
    build_agent_runtime_error_result,
    log_agent_runtime_start,
    prepare_async_agent_runtime_invocation,
)
from app.tools.context import ToolInvocationContext


logger = get_logger("chat_service")

_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_SAFE_TEXT_BLOCK_TYPES = frozenset({"text", "text_delta"})
_STREAM_MODES = ("updates", "messages")


@dataclass(frozen=True)
class AgentRuntimeStreamEvent:
    """Graph 层向 ChatService 暴露的最小流式事件。"""

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


def _namespace_is_direct_agent(namespace: tuple[str, ...]) -> bool:
    """仅允许 create_agent 子图模型文本，显式排除 RAG 子图。"""
    if any(part.split(":", 1)[0] == "rag_subgraph" for part in namespace):
        return False
    return any(
        part.split(":", 1)[0] == "framework_agent"
        for part in namespace
    )


def _safe_message_text(content: object) -> tuple[str, bool]:
    """提取普通文本并返回 block 是否全部通过类型白名单。"""
    if isinstance(content, str):
        return content, True
    if not isinstance(content, list):
        return "", False

    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            return "", False
        if block.get("type") not in _SAFE_TEXT_BLOCK_TYPES:
            return "", False
        text = block.get("text")
        if not isinstance(text, str):
            return "", False
        parts.append(text)
    return "".join(parts), True


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
    if not _namespace_is_direct_agent(namespace):
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

    text, safe_content = _safe_message_text(message.content)
    if isinstance(message, AIMessageChunk):
        projected.append(
            AgentRuntimeStreamEvent(
                event="_message_chunk",
                data={
                    "message_id": message_id,
                    "text": text,
                    "tainted": bool(calls) or not safe_content,
                    "is_last": message.chunk_position == "last",
                },
            )
        )
    elif text and safe_content and not calls:
        projected.append(
            AgentRuntimeStreamEvent(
                event="answer_delta",
                data={"text": text},
                key=f"answer:{message_id}",
            )
        )
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


def _project_update_part(
    data: object,
    namespace: tuple[str, ...],
) -> list[AgentRuntimeStreamEvent]:
    """从 updates 中投影父图阶段、工具完成状态和最终结果。"""
    if not isinstance(data, Mapping):
        return []
    projected: list[AgentRuntimeStreamEvent] = []
    for raw_node_name, raw_update in data.items():
        node_name = str(raw_node_name or "")
        update = raw_update if isinstance(raw_update, Mapping) else {}
        projected.extend(_project_tool_completions(update))

        if namespace:
            continue
        if node_name == "route":
            projected.append(
                AgentRuntimeStreamEvent(
                    event="progress",
                    data={"stage": "routing", "status": "completed"},
                    key="progress:routing:completed",
                )
            )
        elif node_name == "framework_agent" and update.get(
            "pending_rag_result"
        ):
            projected.append(
                AgentRuntimeStreamEvent(
                    event="progress",
                    data={"stage": "rag", "status": "started"},
                    key="progress:rag:started",
                )
            )
        elif node_name == "rag_subgraph":
            projected.append(
                AgentRuntimeStreamEvent(
                    event="progress",
                    data={"stage": "rag", "status": "completed"},
                    key="progress:rag:completed",
                )
            )
        elif node_name == "finalize" and isinstance(
            update.get("result"),
            dict,
        ):
            projected.append(
                AgentRuntimeStreamEvent(
                    event="result",
                    data={"result": update["result"]},
                    key="result:final",
                )
            )
    return projected


def project_graph_stream_part(part: object) -> list[AgentRuntimeStreamEvent]:
    """适配当前 LangGraph v2 part，拒绝未知模式和未知结构。"""
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
        return _project_update_part(part.get("data"), namespace)
    return []


async def _close_async_iterator(iterator: object) -> None:
    """尽最大努力关闭图流，使断开和取消向下传播。"""
    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()


async def stream_agent_langgraph(
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
    """用原生 ``graph.astream`` 执行，并仅产出安全投影与最终结果。"""
    resolved_original_question = (original_question or "").strip()
    resolved_effective_question = (effective_question or "").strip()
    graph_stream: object | None = None
    try:
        invocation = await prepare_async_agent_runtime_invocation(
            top_k=top_k,
            decompose_top_k=decompose_top_k,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original_question,
            effective_question=resolved_effective_question,
            tool_context=tool_context,
        )
        log_agent_runtime_start(
            invocation,
            trace_id=trace_id,
            session_id=session_id,
        )
        graph_stream = invocation.graph.astream(
            invocation.graph_input,
            config=invocation.graph_config,
            context=invocation.tool_context,
            stream_mode=list(_STREAM_MODES),
            subgraphs=True,
            version="v2",
        )
        final_result: dict[str, Any] | None = None
        seen_event_keys: set[str] = set()
        message_buffers: dict[str, list[str]] = {}
        tainted_message_ids: set[str] = set()

        async for part in graph_stream:
            for event in project_graph_stream_part(part):
                if event.event == "_message_chunk":
                    message_id = str(event.data.get("message_id") or "")
                    if event.data.get("tainted"):
                        tainted_message_ids.add(message_id)
                    text = event.data.get("text")
                    if isinstance(text, str) and text:
                        message_buffers.setdefault(message_id, []).append(text)
                    if event.data.get("is_last"):
                        buffered = message_buffers.pop(message_id, [])
                        if message_id not in tainted_message_ids:
                            for index, delta in enumerate(buffered):
                                yield AgentRuntimeStreamEvent(
                                    event="answer_delta",
                                    data={"text": delta},
                                    key=f"answer:{message_id}:{index}",
                                )
                        tainted_message_ids.discard(message_id)
                    continue
                if event.event == "result":
                    final_result = validate_agent_result_payload(
                        event.data.get("result")
                    )
                    continue
                if event.key and event.key in seen_event_keys:
                    continue
                if event.key:
                    seen_event_keys.add(event.key)
                yield event

        if final_result is None:
            raise RuntimeError("LangGraph stream did not produce a valid result")
        logger.info(
            "[trace_id=%s] [agent] langgraph_graph_end status=%s streaming=true",
            trace_id,
            final_result.get("status", ""),
        )
        yield AgentRuntimeStreamEvent(
            event="result",
            data={"result": final_result},
            key="result:final",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "[trace_id=%s] [agent] langgraph_graph_failed streaming=true",
            trace_id,
        )
        result = await asyncio.to_thread(
            build_agent_runtime_error_result,
            write_bad_cases=write_bad_cases,
            bad_case=bad_case,
            trace_id=trace_id,
            session_id=session_id,
            original_question=resolved_original_question,
            effective_question=resolved_effective_question,
            exc=exc,
        )
        yield AgentRuntimeStreamEvent(
            event="result",
            data={"result": result},
            key="result:error",
        )
    finally:
        if graph_stream is not None:
            await _close_async_iterator(graph_stream)


__all__ = [
    "AgentRuntimeStreamEvent",
    "project_graph_stream_part",
    "stream_agent_langgraph",
]
