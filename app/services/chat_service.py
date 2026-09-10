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
from app.storage.memory import append_message_sqlite, get_history_sqlite
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
    initialize_stream_metrics,
    log_chat_process_start,
    log_tool_call_chain,
    record_chat_failure,
    record_stream_completion,
    record_stream_event,
    record_stream_termination,
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
        get_history=get_history_sqlite,
        rewrite_query_with_history=rewrite_query_with_history,
        run_agent=run_agent,
        log_and_write_metrics=log_and_write_metrics,
        write_bad_cases=write_bad_cases,
        append_message=append_message_sqlite,
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
        context = self._initialize_context(request)
        try:
            if not context.original_question:
                return self._empty_question_outcome(context)

            settings = self._deps.settings_provider()
            self._prepare_request(context, settings=settings)
            agent_start = self._deps.perf_counter()
            self._log_agent_start(context)
            try:
                result = self._invoke_agent(context)
            except Exception as exc:
                return self._agent_invocation_exception_outcome(context, exc)
            return self._finish_agent_result(
                context,
                result,
                agent_start=agent_start,
            )
        except Exception as exc:
            return self._service_exception_outcome(context, exc)
        finally:
            self._finalize(context)

    async def stream(
        self,
        request: ChatServiceRequest,
    ) -> AsyncIterator[ChatStreamEvent]:
        """流式编排同一业务流程，并在唯一 finally 中完成收尾。"""
        context = self._initialize_context(request)
        initialize_stream_metrics(context.metrics)
        runtime_stream: object | None = None
        runtime_stream_completed = False
        agent_start: float | None = None
        final_outcome_formed = False
        try:
            if not context.original_question:
                outcome = self._empty_question_outcome(context)
                yield record_stream_event(
                    context,
                    "start",
                    {"session_id_present": bool(context.session_id)},
                    perf_counter=self._deps.perf_counter,
                )
                final_event = self._build_final_stream_event(context, outcome)
                final_outcome_formed = True
                yield final_event
                record_stream_termination(context.metrics, completed=True)
                yield record_stream_event(
                    context,
                    "done",
                    {"reason": "completed"},
                    perf_counter=self._deps.perf_counter,
                )
                return

            yield record_stream_event(
                context,
                "start",
                {"session_id_present": bool(context.session_id)},
                perf_counter=self._deps.perf_counter,
            )
            yield record_stream_event(
                context,
                "progress",
                {"stage": "query_rewrite", "status": "started"},
                perf_counter=self._deps.perf_counter,
            )
            settings = self._deps.settings_provider()
            await asyncio.to_thread(
                self._prepare_request,
                context,
                settings=settings,
            )
            yield record_stream_event(
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
                perf_counter=self._deps.perf_counter,
            )
            yield record_stream_event(
                context,
                "progress",
                {"stage": "routing", "status": "started"},
                perf_counter=self._deps.perf_counter,
            )
            yield record_stream_event(
                context,
                "progress",
                {"stage": "agent", "status": "started"},
                perf_counter=self._deps.perf_counter,
            )

            agent_start = self._deps.perf_counter()
            self._log_agent_start(context)
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
                    yield record_stream_event(
                        context,
                        runtime_event.event,
                        dict(runtime_event.data),
                        perf_counter=self._deps.perf_counter,
                        streamed_answer_parts=streamed_answer_parts,
                    )
                runtime_stream_completed = True
                if result is None:
                    raise RuntimeError(
                        "streaming Runtime did not return a final result"
                    )
            except Exception as exc:
                outcome = await asyncio.to_thread(
                    self._agent_invocation_exception_outcome,
                    context,
                    exc,
                )
            else:
                outcome = self._finish_agent_result(
                    context,
                    result,
                    agent_start=agent_start,
                )

            yield record_stream_event(
                context,
                "progress",
                {"stage": "agent", "status": "completed"},
                perf_counter=self._deps.perf_counter,
            )
            yield record_stream_event(
                context,
                "progress",
                {"stage": "finalizing", "status": "started"},
                perf_counter=self._deps.perf_counter,
            )
            final_answer = str(outcome.result.get("answer") or "")
            record_stream_completion(
                context.metrics,
                streamed_answer_parts=streamed_answer_parts,
                final_answer=final_answer,
            )
            yield record_stream_event(
                context,
                "progress",
                {"stage": "finalizing", "status": "completed"},
                perf_counter=self._deps.perf_counter,
            )
            final_event = self._build_final_stream_event(context, outcome)
            final_outcome_formed = True
            yield final_event
            record_stream_termination(context.metrics, completed=True)
            yield record_stream_event(
                context,
                "done",
                {"reason": "completed"},
                perf_counter=self._deps.perf_counter,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = self._service_exception_outcome(context, exc)
            final_event = self._build_final_stream_event(context, outcome)
            final_outcome_formed = True
            yield final_event
            record_stream_termination(context.metrics, completed=True)
            yield record_stream_event(
                context,
                "done",
                {"reason": "completed"},
                perf_counter=self._deps.perf_counter,
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
                record_stream_termination(context.metrics, completed=False)
            await asyncio.to_thread(self._finalize, context)

    @staticmethod
    async def _close_async_stream(stream: object) -> None:
        """关闭下游异步生成器，使客户端取消尽可能向 Agent 传播。"""
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()

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
        return record_stream_event(
            context,
            "final",
            {"outcome": outcome, "total_cost_ms": total_cost_ms},
            perf_counter=self._deps.perf_counter,
        )

    # 整理问题和会话信息，计算排队耗时并初始化请求上下文及观测记录。
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

    # 为空问题构造 HTTP 400 澄清响应，并记录对应状态和原因。
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

    def _prepare_request(
        self,
        context: ChatRequestContext,
        *,
        settings: AppSettings,
    ) -> None:
        """解析 Runtime 参数，并完成 history/query rewrite 请求准备。"""
        request = context.request
        context.final_top_k = (
            request.top_k if request.top_k is not None else settings.top_k
        )
        context.decompose_top_k = (
            request.decompose_top_k
            if request.decompose_top_k is not None
            else settings.decompose_top_k
        )
        try:
            history = self._deps.get_history(context.session_id)
        except Exception as exc:
            logger.warning(
                "[trace_id=%s] [session_store] history_read_failed "
                "session_id=%s error=%s",
                context.request.trace_id,
                context.session_id,
                str(exc)[:200],
            )
            history = []
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
        context.rewrite_trace = build_query_rewrite_trace(rewrite_result)

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

    @staticmethod
    def _log_agent_start(context: ChatRequestContext) -> None:
        """记录应用层进入唯一 Agent Runtime 的边界。"""
        logger.info(
            "[trace_id=%s] [agent] runtime_start agent_runtime=create_agent",
            context.request.trace_id,
        )

    def _invoke_agent(self, context: ChatRequestContext) -> dict[str, Any]:
        """调用同步 Agent Runtime；Result 解释统一留给收尾入口。"""
        return self._deps.run_agent(
            write_bad_cases=self._deps.write_bad_cases,
            original_question=context.original_question,
            effective_question=context.effective_question,
            top_k=context.final_top_k,
            decompose_top_k=context.decompose_top_k,
            trace_id=context.request.trace_id,
            session_id=context.session_id or None,
            bad_case=context.bad_case,
        )

    def _finish_agent_result(
        self,
        context: ChatRequestContext,
        result: dict[str, Any],
        *,
        agent_start: float,
    ) -> ChatServiceOutcome:
        """统一完成 Result trace、指标、状态策略与 Session 决策。"""
        result = attach_query_rewrite_trace(
            result,
            context.rewrite_trace,
        )
        # 只在 ChatService 副本中合并请求级调用数，保留 Runtime trace 原值。
        result["llm_call_count"] += int(context.metrics["query_rewrite_llm_used"])
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
        trace_metrics = collect_trace_metrics(result)
        policy = get_chat_status_policy(result.get("status", ""))
        apply_result_metrics(
            context.metrics,
            result,
            trace_metrics,
            policy,
        )
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

    def _agent_invocation_exception_outcome(
        self,
        context: ChatRequestContext,
        exc: Exception,
    ) -> ChatServiceOutcome:
        """只处理调用 Runtime 边界本身未转换的基础设施异常。"""
        logger.exception(
            "[trace_id=%s] [answer] Agent Runtime 调用失败 error=%s",
            context.request.trace_id,
            exc,
        )
        record_chat_failure(
            context,
            exc,
            stage="Agent Runtime 调用异常",
            write_bad_cases=self._deps.write_bad_cases,
        )
        return ChatServiceOutcome(
            http_status=500,
            code="MODEL_ERROR",
            message="处理失败，请稍后重试",
            question=context.original_question,
            result={
                **build_internal_error_result(
                    exc,
                    runtime_label="ChatService Agent invocation",
                    answer="",
                    reason="工具调用或模型调用异常",
                    tool_trace=[context.rewrite_trace],
                ),
                "llm_call_count": context.metrics["llm_call_count"],
            },
            response_status="error",
        )

    def _service_exception_outcome(
        self,
        context: ChatRequestContext,
        exc: Exception,
    ) -> ChatServiceOutcome:
        """把请求准备或结果收尾异常映射为脱敏服务级失败。"""
        logger.exception(
            "[trace_id=%s] [chat] 服务流程异常 error=%s",
            context.request.trace_id,
            exc,
        )
        record_chat_failure(
            context,
            exc,
            stage="ChatService 流程异常",
            write_bad_cases=self._deps.write_bad_cases,
        )
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

    # 保存当前轮次的用户消息和助手回复，持久化失败时记录警告。
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

    # 确保请求只收尾一次，按需保存会话并完成指标及日志记录。
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
