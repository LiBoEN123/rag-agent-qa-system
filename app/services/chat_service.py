"""聊天应用服务。

在不依赖 FastAPI 的前提下编排会话改写、Agent Runtime、指标和
会话存储，并返回供 HTTP 层适配的统一结果。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from app.agent.query_rewrite import rewrite_query_with_history
from app.agent.results import (
    build_empty_question_result,
    build_internal_error_result,
)
from app.common.logging import get_logger
from app.config import AppSettings, get_settings
from app.contracts.chat import (
    AgentResultStatus,
    ChatStatusPolicy,
    get_chat_status_policy,
)
from app.agent.runtime import run_agent
from app.agent.streaming import (
    AgentRuntimeStreamEvent,
    stream_agent,
)
from app.observability.metrics import (
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
    log_and_write_metrics,
    now_iso,
    write_bad_cases,
)
from app.storage.session import append_message, get_history
from app.services.chat_models import (
    ChatRequestContext,
    ChatServiceOutcome,
    ChatServiceRequest,
)
from app.services.chat_stream_models import ChatStreamEvent
from app.services.chat_telemetry import (
    apply_result_metrics,
    build_initial_bad_case,
    build_initial_chat_metrics,
    finalize_chat_telemetry,
    log_chat_process_start,
    log_tool_call_chain,
    safe_write_bad_case,
)


logger = get_logger("chat_service")

def _apply_status_policy(
    result: dict[str, Any],
    policy: ChatStatusPolicy,
) -> dict[str, Any]:
    """把唯一状态策略应用到结果副本，不修改 Runtime 原结果。"""
    resolved = dict(result)
    resolved["status"] = policy.result_status.value
    resolved["reason"] = str(
        result.get("reason") or policy.default_reason
    )
    resolved["references"] = list(result.get("references") or [])
    resolved["tool_trace"] = list(result.get("tool_trace") or [])
    if not policy.expose_answer:
        resolved["answer"] = ""
    if not policy.expose_references:
        resolved["references"] = []
    if policy.force_need_clarification is not None:
        resolved["need_clarification"] = (
            policy.force_need_clarification
        )
    if policy.clear_clarification_question:
        resolved["clarification_question"] = ""
    return resolved


@dataclass(frozen=True)
class ChatServiceDependencies:
    """ChatService 的可替换业务边界和副作用依赖。"""

    settings_provider: Callable[[], AppSettings]
    get_history: Callable[[str | None], list[dict]]
    rewrite_query_with_history: Callable[..., dict]
    run_agent: Callable[..., dict[str, Any]]
    build_query_rewrite_trace: Callable[[dict], dict]
    attach_query_rewrite_trace: Callable[[dict, dict], dict]
    collect_trace_metrics: Callable[[dict], dict]
    log_and_write_metrics: Callable[[dict], None]
    write_bad_cases: Callable[[dict], None]
    append_message: Callable[[str | None, str, str], None]
    now_iso: Callable[[], str]
    perf_counter: Callable[[], float]
    stream_agent: Callable[
        ..., AsyncIterator[AgentRuntimeStreamEvent]
    ] = stream_agent


def get_default_chat_service_dependencies() -> ChatServiceDependencies:
    """构造只引用现有函数、且不执行外部调用的默认依赖。"""
    return ChatServiceDependencies(
        settings_provider=get_settings,
        get_history=get_history,
        rewrite_query_with_history=rewrite_query_with_history,
        run_agent=run_agent,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        log_and_write_metrics=log_and_write_metrics,
        write_bad_cases=write_bad_cases,
        append_message=append_message,
        now_iso=now_iso,
        perf_counter=time.perf_counter,
    )


def create_default_chat_service() -> ChatService:
    """创建使用项目默认依赖的聊天应用服务。"""
    return ChatService(get_default_chat_service_dependencies())


class ChatService:
    """以同步或流式方式执行同一套完整聊天业务流程。"""

    def __init__(self, dependencies: ChatServiceDependencies):
        """保存可替换依赖；构造阶段不执行检索、模型或数据库调用。"""
        self._deps = dependencies

    def handle(self, request: ChatServiceRequest) -> ChatServiceOutcome:
        """处理聊天请求，并保证 session 与 metrics 在统一出口收尾。"""
        settings = self._deps.settings_provider()
        context = self._initialize_context(request)
        try:
            if not context.original_question:
                return self._empty_question_outcome(context)

            self._configure_runtime_counts(context, settings=settings)

            self._rewrite_question(context)

            try:
                result = self._run_agent(context)
                self._collect_result_metrics(context, result)
            except Exception as exc:
                return self._runtime_exception_outcome(context, exc)

            return self._build_result_outcome(context, result)
        except Exception as exc:
            return self._unknown_exception_outcome(context, exc)
        finally:
            self._finalize(context)

    async def stream(
        self,
        request: ChatServiceRequest,
    ) -> AsyncIterator[ChatStreamEvent]:
        """流式编排同一业务流程，并在唯一 finally 中完成收尾。"""
        settings = self._deps.settings_provider()
        context = self._initialize_context(request)
        context.metrics.update(
            {
                "streaming": True,
                "stream_event_count": 0,
                "answer_delta_count": 0,
                "streamed_answer_chars": 0,
                "first_event_ms": None,
                "first_answer_delta_ms": None,
                "client_disconnected": True,
                "streamed_answer_matches_final": None,
            }
        )
        runtime_stream: object | None = None
        runtime_stream_completed = False
        agent_start: float | None = None
        final_outcome_formed = False
        try:
            if not context.original_question:
                outcome = self._empty_question_outcome(context)
                yield self._record_stream_event(
                    context,
                    "start",
                    {"session_id_present": bool(context.session_id)},
                )
                final_event = self._build_final_stream_event(context, outcome)
                final_outcome_formed = True
                yield final_event
                context.metrics["client_disconnected"] = False
                yield self._record_stream_event(
                    context,
                    "done",
                    {"reason": "completed"},
                )
                return

            self._configure_runtime_counts(context, settings=settings)
            yield self._record_stream_event(
                context,
                "start",
                {"session_id_present": bool(context.session_id)},
            )
            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "query_rewrite", "status": "started"},
            )
            await asyncio.to_thread(self._rewrite_question, context)
            yield self._record_stream_event(
                context,
                "progress",
                {
                    "stage": "query_rewrite",
                    "status": (
                        "completed"
                        if context.metrics["query_rewrite_used"]
                        else "skipped"
                    ),
                },
            )
            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "routing", "status": "started"},
            )
            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "agent", "status": "started"},
            )

            agent_start = self._deps.perf_counter()
            result: dict[str, Any] | None = None
            streamed_answer_parts: list[str] = []
            try:
                runtime_stream = self._deps.stream_agent(
                    write_bad_cases=self._deps.write_bad_cases,
                    original_question=context.original_question,
                    effective_question=context.effective_question,
                    top_k=context.final_top_k,
                    decompose_top_k=context.decompose_top_k,
                    trace_id=context.request.trace_id,
                    session_id=context.session_id or None,
                    bad_case=context.bad_case,
                )
                async for runtime_event in runtime_stream:
                    if runtime_event.event == "result":
                        runtime_result = runtime_event.data.get("result")
                        if not isinstance(runtime_result, dict):
                            raise RuntimeError(
                                "streaming Runtime returned an invalid result"
                            )
                        result = runtime_result
                        continue
                    if runtime_event.event not in {
                        "progress",
                        "tool",
                        "answer_delta",
                    }:
                        continue
                    if runtime_event.event == "answer_delta":
                        delta = runtime_event.data.get("text")
                        if not isinstance(delta, str) or not delta:
                            continue
                        streamed_answer_parts.append(delta)
                    elif runtime_event.event == "tool" and (
                        runtime_event.data.get("phase") == "completed"
                    ):
                        context.metrics["tool_called"] = True
                        context.metrics["tool_name"] = str(
                            runtime_event.data.get("tool_name") or ""
                        )
                        context.metrics["tool_success"] = (
                            runtime_event.data.get("status") == "success"
                        )
                    yield self._record_stream_event(
                        context,
                        runtime_event.event,
                        dict(runtime_event.data),
                    )
                runtime_stream_completed = True
                if result is None:
                    raise RuntimeError(
                        "streaming Runtime did not return a final result"
                    )
                result = self._complete_agent_result(
                    context,
                    result,
                    agent_start=agent_start,
                )
                self._collect_result_metrics(context, result)
                outcome = self._build_result_outcome(context, result)
            except Exception as exc:
                outcome = await asyncio.to_thread(
                    self._runtime_exception_outcome,
                    context,
                    exc,
                )

            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "agent", "status": "completed"},
            )
            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "finalizing", "status": "started"},
            )
            final_answer = str(outcome.result.get("answer") or "")
            streamed_answer_text = "".join(streamed_answer_parts)
            if streamed_answer_parts:
                context.metrics["streamed_answer_matches_final"] = (
                    streamed_answer_text == final_answer
                )
            yield self._record_stream_event(
                context,
                "progress",
                {"stage": "finalizing", "status": "completed"},
            )
            final_event = self._build_final_stream_event(context, outcome)
            final_outcome_formed = True
            yield final_event
            context.metrics["client_disconnected"] = False
            yield self._record_stream_event(
                context,
                "done",
                {"reason": "completed"},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = self._unknown_exception_outcome(context, exc)
            final_event = self._build_final_stream_event(context, outcome)
            final_outcome_formed = True
            yield final_event
            context.metrics["client_disconnected"] = False
            yield self._record_stream_event(
                context,
                "done",
                {"reason": "completed"},
            )
        finally:
            if runtime_stream is not None and not runtime_stream_completed:
                await self._close_async_stream(runtime_stream)
            if agent_start is not None and not context.metrics.get("agent_cost_ms"):
                context.metrics["agent_cost_ms"] = max(
                    0,
                    int(
                        (self._deps.perf_counter() - agent_start)
                        * 1000
                    ),
                )
            if not final_outcome_formed:
                context.session_should_store = False
                context.metrics["client_disconnected"] = True
                context.metrics["status"] = "client_disconnected"
                context.metrics["reason"] = "client_disconnected"
            await asyncio.to_thread(self._finalize, context)

    @staticmethod
    async def _close_async_stream(stream: object) -> None:
        """关闭下游异步生成器，使客户端取消尽可能向 Agent 传播。"""
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()

    def _configure_runtime_counts(
        self,
        context: ChatRequestContext,
        *,
        settings: AppSettings,
    ) -> None:
        """为同步和流式调用解析同一组 Runtime 数量参数。"""
        request = context.request
        context.final_top_k = (
            request.top_k if request.top_k is not None else settings.top_k
        )
        context.decompose_top_k = (
            request.decompose_top_k
            if request.decompose_top_k is not None
            else settings.decompose_top_k
        )

    def _record_stream_event(
        self,
        context: ChatRequestContext,
        event: str,
        data: dict[str, Any],
    ) -> ChatStreamEvent:
        """更新请求级流指标，并构造不含 HTTP 细节的事件。"""
        metrics = context.metrics
        metrics["stream_event_count"] = int(
            metrics.get("stream_event_count", 0) or 0
        ) + 1
        elapsed_ms = max(
            0,
            int(
                (self._deps.perf_counter() - context.process_start_perf)
                * 1000
            ),
        )
        if metrics.get("first_event_ms") is None:
            metrics["first_event_ms"] = elapsed_ms
        if event == "answer_delta":
            text = str(data.get("text") or "")
            metrics["answer_delta_count"] = int(
                metrics.get("answer_delta_count", 0) or 0
            ) + 1
            metrics["streamed_answer_chars"] = int(
                metrics.get("streamed_answer_chars", 0) or 0
            ) + len(text)
            if metrics.get("first_answer_delta_ms") is None:
                metrics["first_answer_delta_ms"] = elapsed_ms
        return ChatStreamEvent(event=event, data=data)

    def _build_final_stream_event(
        self,
        context: ChatRequestContext,
        outcome: ChatServiceOutcome,
    ) -> ChatStreamEvent:
        """携带待 API 映射的 outcome；该内部对象不会直接 JSON 序列化。"""
        total_cost_ms = max(
            0,
            int(
                (self._deps.perf_counter() - context.request_enter_perf)
                * 1000
            ),
        )
        return self._record_stream_event(
            context,
            "final",
            {"outcome": outcome, "total_cost_ms": total_cost_ms},
        )

    def _initialize_context(
        self,
        request: ChatServiceRequest,
    ) -> ChatRequestContext:
        process_start_perf = self._deps.perf_counter()
        process_start_time = self._deps.now_iso()
        request_enter_perf = (
            request.request_enter_perf
            if request.request_enter_perf is not None
            else process_start_perf
        )
        request_enter_time = request.request_enter_time or process_start_time
        queue_wait_ms = int((process_start_perf - request_enter_perf) * 1000)
        original_question = (request.question or "").strip()
        session_id = (request.session_id or "").strip()

        context = ChatRequestContext(
            request=request,
            process_start_perf=process_start_perf,
            process_start_time=process_start_time,
            request_enter_perf=request_enter_perf,
            original_question=original_question,
            session_id=session_id,
            effective_question=original_question,
            metrics={},
            bad_case={},
        )
        context.metrics = build_initial_chat_metrics(
            context,
            request_enter_time=request_enter_time,
            queue_wait_ms=queue_wait_ms,
        )
        context.bad_case = build_initial_bad_case(
            context,
            created_at=self._deps.now_iso(),
        )
        log_chat_process_start(context, queue_wait_ms=queue_wait_ms)
        return context

    def _empty_question_outcome(
        self,
        context: ChatRequestContext,
    ) -> ChatServiceOutcome:
        logger.warning(
            "[trace_id=%s] [chat] final_status=clarify reason=用户未输入问题",
            context.request.trace_id,
        )
        context.metrics["status"] = "clarify"
        context.metrics["reason"] = "用户未输入问题"
        return ChatServiceOutcome(
            http_status=400,
            code="EMPTY_QUESTION",
            message="问题不能为空，请输入你想查询的内容",
            question=context.original_question,
            result=build_empty_question_result(),
            response_status="clarify",
        )

    def _rewrite_question(self, context: ChatRequestContext) -> None:
        history = self._deps.get_history(context.session_id)
        rewrite_result = self._deps.rewrite_query_with_history(
            question=context.original_question,
            history=history,
            trace_id=context.request.trace_id,
        )
        context.effective_question = (
            rewrite_result.get("rewritten_question")
            or context.original_question
        ).strip() or context.original_question
        rewrite_result["original_question"] = (
            rewrite_result.get("original_question")
            or context.original_question
        )
        rewrite_result["rewritten_question"] = context.effective_question
        context.rewrite_trace = self._deps.build_query_rewrite_trace(
            rewrite_result
        )

        context.metrics["query_rewrite_used"] = bool(
            rewrite_result.get("rewritten", False)
        )
        context.metrics["query_rewrite_llm_used"] = bool(
            rewrite_result.get("rewrite_llm_used", False)
        )
        context.metrics["effective_question_len"] = len(
            context.effective_question
        )
        context.metrics["llm_call_count"] = int(
            context.metrics["query_rewrite_llm_used"]
        )

        logger.info(
            "[trace_id=%s] [query_rewrite] history_count=%s need_check=%s "
            "need_rewrite=%s history_used=%s rewritten=%s "
            "rewrite_llm_used=%s reason=%s original_question=%r "
            "rewritten_question=%r",
            context.request.trace_id,
            len(history),
            rewrite_result.get("need_check", False),
            rewrite_result.get("need_rewrite", False),
            rewrite_result.get("history_used"),
            rewrite_result.get("rewritten"),
            rewrite_result.get("rewrite_llm_used"),
            rewrite_result.get("reason", ""),
            context.original_question,
            context.effective_question,
        )

    def _run_agent(self, context: ChatRequestContext) -> dict[str, Any]:
        agent_start = self._deps.perf_counter()
        logger.info(
            "[trace_id=%s] [agent] runtime_start agent_runtime=create_agent",
            context.request.trace_id,
        )
        result = self._deps.run_agent(
            write_bad_cases=self._deps.write_bad_cases,
            original_question=context.original_question,
            effective_question=context.effective_question,
            top_k=context.final_top_k,
            decompose_top_k=context.decompose_top_k,
            trace_id=context.request.trace_id,
            session_id=context.session_id or None,
            bad_case=context.bad_case,
        )
        return self._complete_agent_result(
            context,
            result,
            agent_start=agent_start,
        )

    def _complete_agent_result(
        self,
        context: ChatRequestContext,
        result: dict[str, Any],
        *,
        agent_start: float,
    ) -> dict[str, Any]:
        """附加 Query Rewrite trace，并记录日志和 Agent 耗时。"""
        result = self._deps.attach_query_rewrite_trace(
            result,
            context.rewrite_trace,
        )
        log_tool_call_chain(
            context.request.trace_id,
            result.get("tool_trace") or [],
        )
        agent_cost_ms = int(
            (self._deps.perf_counter() - agent_start) * 1000
        )
        context.metrics["agent_cost_ms"] = agent_cost_ms
        logger.info(
            "[trace_id=%s] [agent] runtime_end status=%s "
            "agent_cost_ms=%s answer_len=%s reference_count=%s",
            context.request.trace_id,
            result.get("status", ""),
            agent_cost_ms,
            len(result.get("answer", "") or ""),
            len(result.get("references") or []),
        )
        return result

    def _unknown_exception_outcome(
        self,
        context: ChatRequestContext,
        exc: Exception,
    ) -> ChatServiceOutcome:
        """把服务级未知异常映射为现有 INTERNAL_ERROR 结果。"""
        logger.exception(
            "[trace_id=%s] [chat] 未知异常 error=%s",
            context.request.trace_id,
            exc,
        )
        context.metrics["status"] = "internal_error"
        context.metrics["reason"] = str(exc)
        return ChatServiceOutcome(
            http_status=500,
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            question=context.original_question,
            result=build_internal_error_result(
                exc,
                runtime_label="ChatService",
                answer="",
                tool_trace=[],
            ),
            response_status="error",
        )

    def _collect_result_metrics(
        self,
        context: ChatRequestContext,
        result: dict[str, Any],
    ) -> None:
        trace_metrics = self._deps.collect_trace_metrics(result)
        policy = get_chat_status_policy(result.get("status", ""))
        apply_result_metrics(
            context.metrics,
            result,
            trace_metrics,
            policy,
        )

    def _runtime_exception_outcome(
        self,
        context: ChatRequestContext,
        exc: Exception,
    ) -> ChatServiceOutcome:
        logger.exception(
            "[trace_id=%s] [answer] 处理失败 error=%s",
            context.request.trace_id,
            exc,
        )
        context.metrics["status"] = "internal_error"
        context.metrics["action"] = ""
        context.metrics["tool_name"] = ""
        context.metrics["tool_called"] = False
        context.metrics["tool_success"] = False
        context.metrics["router_llm_used"] = False
        context.metrics["llm_call_count"] = int(
            context.metrics["query_rewrite_llm_used"]
        )
        context.metrics["reason"] = str(exc)
        context.bad_case["route"] = ""
        context.bad_case["error_type"] = "internal_error"
        context.bad_case["note"] = f"Agent Runtime 执行异常：{exc}"
        safe_write_bad_case(
            context.bad_case,
            trace_id=context.request.trace_id,
            write_bad_cases=self._deps.write_bad_cases,
        )
        return ChatServiceOutcome(
            http_status=500,
            code="MODEL_ERROR",
            message="处理失败，请稍后重试",
            question=context.original_question,
            result=build_internal_error_result(
                exc,
                runtime_label="ChatService Agent invocation",
                answer="",
                reason="工具调用或模型调用异常",
                tool_trace=[context.rewrite_trace],
            ),
            response_status="error",
        )

    def _build_result_outcome(
        self,
        context: ChatRequestContext,
        result: dict[str, Any],
    ) -> ChatServiceOutcome:
        policy = get_chat_status_policy(result.get("status", ""))
        resolved_result = _apply_status_policy(result, policy)
        log = (
            logger.info
            if policy.result_status is AgentResultStatus.OK
            else logger.warning
        )
        log(
            "[trace_id=%s] [chat] final_status=%s reason=%s",
            context.request.trace_id,
            policy.log_status,
            resolved_result["reason"],
        )
        context.session_should_store = True
        context.session_assistant_content = (
            resolved_result["answer"]
            or resolved_result["clarification_question"]
            or resolved_result["reason"]
        )
        return ChatServiceOutcome(
            http_status=policy.http_status,
            code=policy.code,
            message=policy.message,
            question=context.original_question,
            result=resolved_result,
            response_status=policy.public_status,
        )

    def _save_session_turn(self, context: ChatRequestContext) -> None:
        if not context.session_id:
            return
        try:
            self._deps.append_message(
                context.session_id,
                "user",
                context.original_question,
            )
            self._deps.append_message(
                context.session_id,
                "assistant",
                context.session_assistant_content,
            )
        except Exception as exc:
            logger.warning(
                "[trace_id=%s] [session_store] append_failed "
                "session_id=%s error=%s",
                context.request.trace_id,
                context.session_id,
                str(exc)[:200],
            )

    def _finalize(self, context: ChatRequestContext) -> None:
        if context.finalized:
            return
        context.finalized = True
        if context.session_should_store:
            self._save_session_turn(context)
        finalize_chat_telemetry(
            context,
            now_iso=self._deps.now_iso,
            perf_counter=self._deps.perf_counter,
            log_and_write_metrics=self._deps.log_and_write_metrics,
        )

__all__ = [
    "ChatService",
    "ChatServiceDependencies",
    "create_default_chat_service",
    "get_default_chat_service_dependencies",
]
