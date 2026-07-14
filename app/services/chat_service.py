"""聊天应用服务。

在不依赖 FastAPI 的前提下编排会话改写、路由、RAG 就绪检查、Agent
Runtime、指标和会话存储，并返回供 HTTP 层适配的统一结果。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.agent import route_agent_action
from app.agent.query_rewrite import rewrite_query_with_history
from app.agent.runtime import run_agent_v2
from app.common.logging import get_logger
from app.config import Settings, validate_agent_runtime
from app.observability.metrics import (
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
    log_and_write_metrics,
    normalize_metrics_status,
    now_iso,
    write_bad_cases,
)
from app.rag import get_rag_status
from app.storage.session import append_message, get_history


logger = get_logger("chat_service")

RAG_READY_ACTIONS = {
    "search_knowledge_base",
    "list_documents",
    "get_chunk_text",
}

_TIMING_METRIC_KEYS = (
    "retrieve_cost_ms",
    "decompose_cost_ms",
    "rerank_cost_ms",
    "retrieve_with_sub_queries_cost_ms",
    "context_build_cost_ms",
    "llm_cost_ms",
    "parse_cost_ms",
    "rag_answer_cost_ms",
)

CHAT_RESULT_RESPONSE_MAP = {
    "MODEL_ERROR": {
        "code": "MODEL_ERROR",
        "message": "模型调用失败",
        "status": "error",
        "default_reason": "模型调用失败",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "error",
    },
    "NEED_CLARIFICATION": {
        "code": "NEED_CLARIFICATION",
        "message": "问题信息不足，需要补充后再回答",
        "status": "clarify",
        "default_reason": "问题缺少关键限定信息",
        "answer": "",
        "references": [],
        "need_clarification": True,
        "log_status": "need_clarification",
    },
    "NO_RELEVANT_INFO": {
        "code": "NO_RELEVANT_INFO",
        "message": "我没有在文档中找到相关信息",
        "status": "no_evidence",
        "default_reason": "模型判断当前上下文不足以支撑回答",
        "clarification_question": "",
        "log_status": "no_evidence",
    },
    "TOOL_ERROR": {
        "code": "TOOL_ERROR",
        "message": "工具调用失败",
        "status": "error",
        "default_reason": "工具调用失败",
        "references": [],
        "log_status": "error",
    },
    "OUT_OF_SCOPE": {
        "code": "OUT_OF_SCOPE",
        "message": "问题超出当前知识库范围",
        "status": "out_of_scope",
        "default_reason": "问题超出知识库范围",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "out_of_scope",
    },
    "MODEL_BAD_OUTPUT": {
        "code": "MODEL_BAD_OUTPUT",
        "message": "模型输出格式异常，请稍候重试",
        "status": "error",
        "default_reason": "模型返回结构不符合预期",
        "references": [],
        "clarification_question": "",
        "log_status": "error",
    },
    "INTERNAL_ERROR": {
        "code": "INTERNAL_ERROR",
        "message": "服务器内部错误",
        "status": "error",
        "default_reason": "Agent 内部执行异常",
        "references": [],
        "need_clarification": False,
        "clarification_question": "",
        "log_status": "error",
    },
}


@dataclass(frozen=True)
class ChatServiceRequest:
    """一次聊天业务调用所需的框架无关输入。"""

    question: str
    top_k: int | None
    decompose_top_k: int | None
    session_id: str
    trace_id: str
    request_enter_time: str = ""
    request_enter_perf: float | None = None


@dataclass(frozen=True)
class ChatServiceOutcome:
    """聊天业务结果及其 HTTP 适配元数据。"""

    http_status: int
    code: int | str
    message: str
    question: str
    result: dict[str, Any]
    response_status: str = "ok"
    response_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ChatServiceDependencies:
    """ChatService 的可替换业务边界和副作用依赖。"""

    settings: Any
    get_history: Callable[[str | None], list[dict]]
    rewrite_query_with_history: Callable[..., dict]
    route_agent_action: Callable[..., dict]
    get_rag_status: Callable[[], dict]
    run_agent_v2: Callable[..., dict[str, Any]]
    run_agent_langgraph: Callable[..., dict[str, Any]]
    validate_agent_runtime: Callable[[str | None], str]
    build_query_rewrite_trace: Callable[[dict], dict]
    attach_query_rewrite_trace: Callable[[dict, dict], dict]
    collect_trace_metrics: Callable[[dict], dict]
    normalize_metrics_status: Callable[[str], str]
    log_and_write_metrics: Callable[[dict], None]
    write_bad_cases: Callable[[dict], None]
    append_message: Callable[[str | None, str, str], None]
    now_iso: Callable[[], str]
    perf_counter: Callable[[], float]
    response_map: Mapping[str, dict[str, Any]]


@dataclass
class _ChatContext:
    """单次 handle 调用内部共享的可变处理上下文。"""

    request: ChatServiceRequest
    process_start_perf: float
    process_start_time: str
    request_enter_perf: float
    original_question: str
    session_id: str
    effective_question: str
    metrics: dict[str, Any]
    bad_case: dict[str, Any]
    final_top_k: int = 0
    decompose_top_k: int = 0
    rewrite_trace: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    session_should_store: bool = False
    session_assistant_content: str = ""


def _run_agent_langgraph(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """延迟导入 LangGraph Runtime，保持 legacy 启动路径轻量。"""
    from app.graph.runtime import run_agent_langgraph

    return run_agent_langgraph(*args, **kwargs)


def get_default_chat_service_dependencies() -> ChatServiceDependencies:
    """构造只引用现有函数、且不执行外部调用的默认依赖。"""
    return ChatServiceDependencies(
        settings=Settings,
        get_history=get_history,
        rewrite_query_with_history=rewrite_query_with_history,
        route_agent_action=route_agent_action,
        get_rag_status=get_rag_status,
        run_agent_v2=run_agent_v2,
        run_agent_langgraph=_run_agent_langgraph,
        validate_agent_runtime=validate_agent_runtime,
        build_query_rewrite_trace=build_query_rewrite_trace,
        attach_query_rewrite_trace=attach_query_rewrite_trace,
        collect_trace_metrics=collect_trace_metrics,
        normalize_metrics_status=normalize_metrics_status,
        log_and_write_metrics=log_and_write_metrics,
        write_bad_cases=write_bad_cases,
        append_message=append_message,
        now_iso=now_iso,
        perf_counter=time.perf_counter,
        response_map=CHAT_RESULT_RESPONSE_MAP,
    )


def create_default_chat_service() -> ChatService:
    """创建使用项目默认依赖的聊天应用服务。"""
    return ChatService(get_default_chat_service_dependencies())


class ChatService:
    """同步执行一次完整聊天业务流程。"""

    def __init__(self, dependencies: ChatServiceDependencies):
        """保存可替换依赖；构造阶段不执行检索、模型或数据库调用。"""
        self._deps = dependencies

    def handle(self, request: ChatServiceRequest) -> ChatServiceOutcome:
        """处理聊天请求，并保证 session 与 metrics 在统一出口收尾。"""
        context = self._initialize_context(request)
        try:
            if not context.original_question:
                return self._empty_question_outcome(context)

            context.final_top_k = (
                request.top_k
                if request.top_k is not None
                else self._deps.settings.TOP_K
            )
            context.decompose_top_k = (
                request.decompose_top_k
                if request.decompose_top_k is not None
                else self._deps.settings.DECOMPOSE_TOP_K
            )

            self._rewrite_question(context)
            self._route_question(context)

            not_ready_outcome = self._check_rag_ready(context)
            if not_ready_outcome is not None:
                return not_ready_outcome

            try:
                result = self._run_agent(context)
                self._collect_result_metrics(context, result)
            except Exception as exc:
                return self._runtime_exception_outcome(context, exc)

            return self._build_result_outcome(context, result)
        except Exception as exc:
            logger.exception(
                "[trace_id=%s] [chat] 未知异常 error=%s",
                request.trace_id,
                exc,
            )
            context.metrics["status"] = "internal_error"
            context.metrics["reason"] = str(exc)
            return ChatServiceOutcome(
                http_status=500,
                code="INTERNAL_ERROR",
                message="服务器内部错误",
                question=context.original_question,
                result=self._response_result(),
                # 保留重构前未知异常分支未显式覆盖的默认 API status。
                response_status="ok",
            )
        finally:
            self._finalize(context)

    def _initialize_context(self, request: ChatServiceRequest) -> _ChatContext:
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

        logger.info(
            "[trace_id=%s] [chat] process_start queue_wait_ms=%s "
            "question_len=%s session_id=%s preview=%s",
            request.trace_id,
            queue_wait_ms,
            len(original_question),
            session_id or "-",
            original_question[:50],
        )

        metrics = {
            "trace_id": request.trace_id,
            "request_enter_time": request_enter_time,
            "process_start_time": process_start_time,
            "end_time": "",
            "queue_wait_ms": queue_wait_ms,
            "latency_ms": 0,
            "total_cost_ms": 0,
            "route_cost_ms": 0,
            "rag_ready_check_cost_ms": 0,
            "agent_cost_ms": 0,
            "retrieve_cost_ms": 0,
            "decompose_cost_ms": 0,
            "rerank_cost_ms": 0,
            "retrieve_with_sub_queries_cost_ms": 0,
            "context_build_cost_ms": 0,
            "llm_cost_ms": 0,
            "parse_cost_ms": 0,
            "rag_answer_cost_ms": 0,
            "status": "internal_error",
            "question_len": len(original_question),
            "reference_count": 0,
            "retrieved_count": 0,
            "tool_called": False,
            "tool_success": False,
            "tool_name": "",
            "router_llm_used": False,
            "decompose_llm_used": False,
            "answer_llm_used": False,
            "llm_call_count": 0,
            "query_rewrite_used": False,
            "query_rewrite_llm_used": False,
            "effective_question_len": len(original_question),
            "action": "",
            "reason": "",
        }
        bad_case = {
            "created_at": self._deps.now_iso(),
            "trace_id": request.trace_id,
            "question": original_question,
            "answer": "",
            "references": [],
            "route": "",
            "error_type": "",
            "note": "",
        }
        return _ChatContext(
            request=request,
            process_start_perf=process_start_perf,
            process_start_time=process_start_time,
            request_enter_perf=request_enter_perf,
            original_question=original_question,
            session_id=session_id,
            effective_question=original_question,
            metrics=metrics,
            bad_case=bad_case,
        )

    def _empty_question_outcome(
        self,
        context: _ChatContext,
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
            result=self._response_result(
                need_clarification=True,
                clarification_question=(
                    "请直接输入你想查询的具体问题，例如：年假申请流程是什么？"
                ),
                reason="用户未输入问题",
            ),
            response_status="clarify",
        )

    def _rewrite_question(self, context: _ChatContext) -> None:
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

    def _route_question(self, context: _ChatContext) -> None:
        route_start = self._deps.perf_counter()
        context.decision = self._deps.route_agent_action(
            context.effective_question,
            context.final_top_k,
            trace_id=context.request.trace_id,
        )
        route_cost_ms = int(
            (self._deps.perf_counter() - route_start) * 1000
        )
        context.metrics["route_cost_ms"] = route_cost_ms
        context.action = context.decision.get("action") or ""
        context.metrics["router_llm_used"] = bool(
            context.decision.get(
                "router_llm_used",
                context.decision.get("router_source")
                in {"llm", "rule_based_fallback"},
            )
        )
        context.metrics["llm_call_count"] = (
            int(context.metrics["query_rewrite_llm_used"])
            + int(context.metrics["router_llm_used"])
        )
        logger.info(
            "[trace_id=%s] [agent] route_end action=%s route_cost_ms=%s "
            "reason=%s",
            context.request.trace_id,
            context.action,
            route_cost_ms,
            context.decision.get("reason", ""),
        )
        logger.info(
            "[trace_id=%s] [agent] route_final action=%s",
            context.request.trace_id,
            context.action,
        )

    def _check_rag_ready(
        self,
        context: _ChatContext,
    ) -> ChatServiceOutcome | None:
        if context.action not in RAG_READY_ACTIONS:
            return None

        ready_check_start = self._deps.perf_counter()
        rag_status = self._deps.get_rag_status()
        ready_check_cost_ms = int(
            (self._deps.perf_counter() - ready_check_start) * 1000
        )
        context.metrics["rag_ready_check_cost_ms"] = ready_check_cost_ms
        logger.info(
            "[trace_id=%s] [rag] ready_check_end ready=%s "
            "rag_ready_check_cost_ms=%s message=%s",
            context.request.trace_id,
            rag_status.get("ready"),
            ready_check_cost_ms,
            rag_status.get("message", ""),
        )
        if rag_status["ready"]:
            return None

        logger.warning(
            "[trace_id=%s] [chat] final_status=empty_index "
            "reason=知识库未准备完成",
            context.request.trace_id,
        )
        context.metrics["status"] = "empty_index"
        context.metrics["action"] = context.action
        context.metrics["reason"] = "知识库未准备完成"
        context.bad_case["route"] = context.action or "rag"
        context.bad_case["error_type"] = "index_not_ready"
        context.bad_case["note"] = "知识库未准备完成，请检查知识库"
        self._write_bad_case(context.bad_case, context.request.trace_id)

        context.session_should_store = True
        context.session_assistant_content = (
            rag_status["message"] or "知识库未准备完成"
        )
        return ChatServiceOutcome(
            http_status=503,
            code="NO_DOCUMENTS",
            message=rag_status["message"],
            question=context.original_question,
            result=self._response_result(
                reason="知识库未准备完成",
                tool_trace=[context.rewrite_trace],
            ),
            response_status="empty_index",
        )

    def _run_agent(self, context: _ChatContext) -> dict[str, Any]:
        agent_start = self._deps.perf_counter()
        runtime_name = self._deps.validate_agent_runtime(
            self._deps.settings.AGENT_RUNTIME
        )
        runtime_runner = (
            self._deps.run_agent_langgraph
            if runtime_name == "langgraph"
            else self._deps.run_agent_v2
        )
        result = runtime_runner(
            write_bad_cases=self._deps.write_bad_cases,
            question=context.effective_question,
            top_k=context.final_top_k,
            decompose_top_k=context.decompose_top_k,
            decision=context.decision,
            trace_id=context.request.trace_id,
            session_id=context.session_id or None,
            bad_case=context.bad_case,
        )
        result = self._deps.attach_query_rewrite_trace(
            result,
            context.rewrite_trace,
        )
        agent_cost_ms = int(
            (self._deps.perf_counter() - agent_start) * 1000
        )
        context.metrics["agent_cost_ms"] = agent_cost_ms
        logger.info(
            "[trace_id=%s] [agent] runtime_end agent_runtime=%s status=%s "
            "agent_cost_ms=%s answer_len=%s reference_count=%s",
            context.request.trace_id,
            runtime_name,
            result.get("status", ""),
            agent_cost_ms,
            len(result.get("answer", "") or ""),
            len(result.get("references") or []),
        )
        return result

    def _collect_result_metrics(
        self,
        context: _ChatContext,
        result: dict[str, Any],
    ) -> None:
        trace_metrics = self._deps.collect_trace_metrics(result)
        context.metrics["status"] = self._deps.normalize_metrics_status(
            result.get("status", "")
        )
        context.metrics["reference_count"] = len(
            result.get("references") or []
        )
        for key in (
            "retrieved_count",
            "tool_called",
            "tool_success",
            "tool_name",
            "router_llm_used",
            "decompose_llm_used",
            "answer_llm_used",
            "llm_call_count",
        ):
            context.metrics[key] = trace_metrics[key]
        timings = trace_metrics["timings"]
        for timing_key in _TIMING_METRIC_KEYS:
            context.metrics[timing_key] = int(
                (timings.get(timing_key, 0) or 0)
            )
        context.metrics["action"] = result.get(
            "action",
            context.action,
        )
        context.metrics["reason"] = result.get("reason", "")

    def _runtime_exception_outcome(
        self,
        context: _ChatContext,
        exc: Exception,
    ) -> ChatServiceOutcome:
        logger.exception(
            "[trace_id=%s] [answer] 处理失败 error=%s",
            context.request.trace_id,
            exc,
        )
        context.metrics["status"] = "internal_error"
        context.metrics["action"] = context.action
        context.metrics["tool_name"] = context.decision.get("tool_name", "")
        context.metrics["tool_called"] = bool(
            context.decision.get("tool_name", "")
        )
        context.metrics["tool_success"] = False
        context.metrics["router_llm_used"] = bool(
            context.decision.get(
                "router_llm_used",
                context.decision.get("router_source")
                in {"llm", "rule_based_fallback"},
            )
        )
        context.metrics["llm_call_count"] = (
            int(context.metrics["query_rewrite_llm_used"])
            + int(context.metrics["router_llm_used"])
        )
        context.metrics["reason"] = str(exc)
        context.bad_case["route"] = (
            context.action or context.decision.get("decision", "")
        )
        context.bad_case["error_type"] = "internal_error"
        context.bad_case["note"] = f"Agent Runtime 执行异常：{exc}"
        self._write_bad_case(context.bad_case, context.request.trace_id)
        return ChatServiceOutcome(
            http_status=500,
            code="MODEL_ERROR",
            message="处理失败，请稍后重试",
            question=context.original_question,
            result=self._response_result(
                reason="工具调用或模型调用异常",
                tool_trace=[context.rewrite_trace],
            ),
            response_status="error",
        )

    def _build_result_outcome(
        self,
        context: _ChatContext,
        result: dict[str, Any],
    ) -> ChatServiceOutcome:
        response_config = self._deps.response_map.get(result["status"])
        if response_config is not None:
            logger.warning(
                "[trace_id=%s] [chat] final_status=%s reason=%s",
                context.request.trace_id,
                response_config["log_status"],
                result.get("reason", response_config["default_reason"]),
            )
            context.session_should_store = True
            context.session_assistant_content = (
                response_config.get("answer", result.get("answer", ""))
                or result.get("clarification_question", "")
                or result.get("reason", response_config["default_reason"])
            )
            return ChatServiceOutcome(
                http_status=200,
                code=response_config["code"],
                message=response_config["message"],
                question=context.original_question,
                result=result,
                response_status=response_config["status"],
                response_config=response_config,
            )

        logger.info(
            "[trace_id=%s] [chat] final_status=ok reason=%s",
            context.request.trace_id,
            result.get("reason", "正常回答"),
        )
        context.session_should_store = True
        context.session_assistant_content = (
            result["answer"] or result.get("reason", "正常回答")
        )
        return ChatServiceOutcome(
            http_status=200,
            code=0,
            message="success",
            question=context.original_question,
            result=result,
            response_status="ok",
        )

    def _write_bad_case(self, record: dict, trace_id: str) -> None:
        try:
            self._deps.write_bad_cases(record)
        except Exception as exc:
            logger.warning(
                "[trace_id=%s] [bad_case] write_failed error=%s",
                trace_id,
                str(exc)[:200],
            )

    def _save_session_turn(self, context: _ChatContext) -> None:
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

    def _finalize(self, context: _ChatContext) -> None:
        process_end_perf = self._deps.perf_counter()
        context.metrics["end_time"] = self._deps.now_iso()
        context.metrics["latency_ms"] = int(
            (process_end_perf - context.process_start_perf) * 1000
        )
        context.metrics["total_cost_ms"] = int(
            (process_end_perf - context.request_enter_perf) * 1000
        )
        if context.session_should_store:
            self._save_session_turn(context)

        logger.info(
            "[trace_id=%s] [chat] process_end status=%s queue_wait_ms=%s "
            "latency_ms=%s total_cost_ms=%s route_cost_ms=%s "
            "rag_ready_check_cost_ms=%s agent_cost_ms=%s "
            "retrieve_cost_ms=%s decompose_cost_ms=%s rerank_cost_ms=%s "
            "llm_cost_ms=%s llm_call_count=%s query_rewrite_used=%s "
            "query_rewrite_llm_used=%s effective_question_len=%s",
            context.request.trace_id,
            context.metrics.get("status"),
            context.metrics.get("queue_wait_ms"),
            context.metrics.get("latency_ms"),
            context.metrics.get("total_cost_ms"),
            context.metrics.get("route_cost_ms"),
            context.metrics.get("rag_ready_check_cost_ms"),
            context.metrics.get("agent_cost_ms"),
            context.metrics.get("retrieve_cost_ms"),
            context.metrics.get("decompose_cost_ms"),
            context.metrics.get("rerank_cost_ms"),
            context.metrics.get("llm_cost_ms"),
            context.metrics.get("llm_call_count"),
            context.metrics.get("query_rewrite_used"),
            context.metrics.get("query_rewrite_llm_used"),
            context.metrics.get("effective_question_len"),
        )
        try:
            self._deps.log_and_write_metrics(context.metrics)
        except Exception as exc:
            logger.exception(
                "[trace_id=%s] [metrics] write_failed error=%s",
                context.request.trace_id,
                exc,
            )

    @staticmethod
    def _response_result(
        *,
        answer: str = "",
        references: list | None = None,
        retrieval_confidence: float = 0.0,
        answer_confidence: float = 0.0,
        need_clarification: bool = False,
        clarification_question: str = "",
        reason: str = "",
        action: str = "",
        tool_result_summary: str = "",
        tool_trace: list | None = None,
    ) -> dict[str, Any]:
        """构造非 Runtime 分支供现有 Response Builder 消费的字段。"""
        return {
            "answer": answer,
            "references": references or [],
            "retrieval_confidence": retrieval_confidence,
            "answer_confidence": answer_confidence,
            "need_clarification": need_clarification,
            "clarification_question": clarification_question,
            "reason": reason,
            "action": action,
            "tool_result_summary": tool_result_summary,
            "tool_trace": tool_trace or [],
        }


__all__ = [
    "ChatService",
    "ChatServiceDependencies",
    "ChatServiceOutcome",
    "ChatServiceRequest",
    "CHAT_RESULT_RESPONSE_MAP",
    "RAG_READY_ACTIONS",
    "create_default_chat_service",
    "get_default_chat_service_dependencies",
]
