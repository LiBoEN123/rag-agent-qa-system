"""FastAPI chat controller and application assembly."""

from fastapi import FastAPI, HTTPException, Response, Request
from contextlib import asynccontextmanager
import time
from app.rag import (
    preview_docs,
    preview_chunks,
    search_relevant_chunks,
    rerank,
    get_rag_status,
    needs_rebuild,
    load_index_cache,
    index_cache_exists,
    rebuild_and_persist_index,
)
from app.config import Settings
from app.common.logging import get_logger
from app.agent import route_agent_action
from app.agent.runtime import run_agent_v2
from app.observability import create_trace_id
from app.agent.query_rewrite import rewrite_query_with_history
from app.rag.retrieval import safe_float, should_refuse
from app.storage.session import append_message, get_history
from app.observability.metrics import (
    BAD_CASES_PATH,
    EXPECTED_NON_BAD_CASE_TYPES,
    METRICS_PATH,
    REAL_BAD_CASE,
    attach_query_rewrite_trace,
    build_query_rewrite_trace,
    collect_trace_metrics,
    extract_llm_usage,
    log_and_write_metrics,
    normalize_metrics_status,
    now_iso,
    read_recent_metrics,
    should_write_bad_case,
    write_bad_cases,
    write_metrics,
)
from app.observability.middleware import trace_and_timing_middleware
from app.api.diagnostics import (
    DiagnosticsDependencies,
    create_diagnostics_router,
)
from app.api.responses import CHAT_RESULT_RESPONSE_MAP, ChatResponseBuilder
from app.api.lifecycle import StartupDependencies, initialize_rag
from app.api.schemas import (
    ChatData,
    ChatReferences,
    ChatRequest,
    ChatResponse,
    PublicStatus,
    SearchData,
    SearchResponse,
    SearchResult,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_on_event()
    yield


app = FastAPI(title='Mini Rag API', version='1.0.0', lifespan=lifespan)
logger = get_logger('chat_service')

app.middleware("http")(trace_and_timing_middleware)

_chat_response_builder = ChatResponseBuilder(
    response_type=ChatResponse,
    data_type=ChatData,
    safe_float=safe_float,
    extract_llm_usage=extract_llm_usage,
)
build_chat_response = _chat_response_builder.build
build_chat_response_from_result = _chat_response_builder.build_from_result


def save_session_turn(
        session_id: str,
        user_question: str,
        assistant_content: str,
        trace_id: str = "",
) -> None:
    """保存一轮成功生成的会话消息，失败只记日志不影响响应。"""
    if not session_id:
        return

    try:
        append_message(session_id, "user", user_question)
        append_message(session_id, "assistant", assistant_content)
    except Exception as e:
        logger.warning(
            f"[trace_id={trace_id}] [session_store] append_failed "
            f"session_id={session_id} error={str(e)[:200]}"
        )


# =========================
# FastAPI 路由
# =========================

def start_on_event():
    """FastAPI 启动时校验配置并加载或重建 RAG 索引。"""
    initialize_rag(
        StartupDependencies(
            settings=Settings,
            needs_rebuild=needs_rebuild,
            index_cache_exists=index_cache_exists,
            load_index_cache=load_index_cache,
            rebuild_and_persist_index=rebuild_and_persist_index,
        )
    )

@app.get("/health")
def health():
    """返回服务健康检查结果。"""
    return {
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok"
  }
}

@app.post("/chat", response_model=ChatResponse)
def ask_api(req: ChatRequest, request: Request, response: Response):
    """处理 /chat 请求，串联校验、路由、Agent 执行、指标和响应构造。"""
    process_start_perf = time.perf_counter()
    process_start_time = now_iso()

    trace_id = getattr(request.state, "trace_id", "") or create_trace_id()

    request_enter_perf = getattr(
        request.state,
        "request_enter_perf",
        process_start_perf
    )

    request_enter_time = getattr(
        request.state,
        "request_enter_time",
        process_start_time
    )

    queue_wait_ms = int((process_start_perf - request_enter_perf) * 1000)

    original_question = (req.question or "").strip()
    session_id = (req.session_id or "").strip()
    effective_question = original_question
    session_should_store = False
    session_assistant_content = ""

    logger.info(
        f"[trace_id={trace_id}] [chat] process_start "
        f"queue_wait_ms={queue_wait_ms} "
        f"question_len={len(original_question)} "
        f"session_id={session_id or '-'} "
        f"preview={original_question[:50]}"
    )

    bad_case = {
        "created_at": now_iso(),
        "trace_id":trace_id,
        "question":original_question,
        "answer":"",
        "references":[],
        "route":"",
        "error_type":"",
        "note":""
    }

    metrics = {
        "trace_id": trace_id,

        # HTTP 层进入时间
        "request_enter_time": request_enter_time,

        # /chat 真正开始处理时间
        "process_start_time": process_start_time,

        "end_time": "",

        # 进入服务后等待进入处理函数的时间
        "queue_wait_ms": queue_wait_ms,

        # /chat 函数内部处理耗时
        "latency_ms": 0,

        # request_enter 到 response_end 的总耗时
        "total_cost_ms": 0,

        # 细分阶段耗时
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
        "effective_question_len": len(effective_question),
        "action": "",
        "reason": "",
    }

    try:
        if not original_question:
            logger.warning(f"[trace_id={trace_id}] [chat] final_status=clarify reason=用户未输入问题")
            response.status_code = 400

            metrics["status"] = "clarify"
            metrics["reason"] = "用户未输入问题"

            return build_chat_response(
                code="EMPTY_QUESTION",
                message="问题不能为空，请输入你想查询的内容",
                trace_id=trace_id,
                question=original_question,
                need_clarification=True,
                status="clarify",
                clarification_question="请直接输入你想查询的具体问题，例如：年假申请流程是什么？",
                reason="用户未输入问题"
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K
        decompose_top_k = req.decompose_top_k if req.decompose_top_k is not None else Settings.DECOMPOSE_TOP_K

        history = get_history(session_id)
        rewrite_result = rewrite_query_with_history(
            question=original_question,
            history=history,
            trace_id=trace_id,
        )
        effective_question = (
            rewrite_result.get("rewritten_question")
            or original_question
        ).strip() or original_question
        rewrite_result["original_question"] = (
            rewrite_result.get("original_question")
            or original_question
        )
        rewrite_result["rewritten_question"] = effective_question
        rewrite_trace = build_query_rewrite_trace(rewrite_result)

        metrics["query_rewrite_used"] = bool(rewrite_result.get("rewritten", False))
        metrics["query_rewrite_llm_used"] = bool(rewrite_result.get("rewrite_llm_used", False))
        metrics["effective_question_len"] = len(effective_question)
        metrics["llm_call_count"] = int(metrics["query_rewrite_llm_used"])

        logger.info(
            f"[trace_id={trace_id}] [query_rewrite] "
            f"history_count={len(history)} "
            f"need_check={rewrite_result.get('need_check', False)} "
            f"need_rewrite={rewrite_result.get('need_rewrite', False)} "
            f"history_used={rewrite_result.get('history_used')} "
            f"rewritten={rewrite_result.get('rewritten')} "
            f"rewrite_llm_used={rewrite_result.get('rewrite_llm_used')} "
            f"reason={rewrite_result.get('reason', '')} "
            f"original_question={original_question!r} "
            f"rewritten_question={effective_question!r}"
        )

        route_start = time.perf_counter()

        decision = route_agent_action(effective_question, final_top_k, trace_id=trace_id)

        route_cost_ms = int((time.perf_counter() - route_start) * 1000)
        metrics["route_cost_ms"] = route_cost_ms

        action = decision.get("action")
        metrics["router_llm_used"] = bool(
            decision.get(
                "router_llm_used",
                decision.get("router_source") in {"llm", "rule_based_fallback"}
            )
        )
        metrics["llm_call_count"] = (
            int(metrics["query_rewrite_llm_used"])
            + int(metrics["router_llm_used"])
        )

        logger.info(
            f"[trace_id={trace_id}] [agent] route_end "
            f"action={action} "
            f"route_cost_ms={route_cost_ms} "
            f"reason={decision.get('reason', '')}"
        )

        logger.info(
            f"[trace_id={trace_id}] [agent] route_final action={action}"
        )

        need_rag_ready_actions = {
            "search_knowledge_base",
            "list_documents",
            "get_chunk_text"
        }

        if action in need_rag_ready_actions:
            ready_check_start = time.perf_counter()

            rag_status = get_rag_status()

            rag_ready_check_cost_ms = int(
                (time.perf_counter() - ready_check_start) * 1000
            )

            metrics["rag_ready_check_cost_ms"] = rag_ready_check_cost_ms

            logger.info(
                f"[trace_id={trace_id}] [rag] ready_check_end "
                f"ready={rag_status.get('ready')} "
                f"rag_ready_check_cost_ms={rag_ready_check_cost_ms} "
                f"message={rag_status.get('message', '')}"
            )

            if not rag_status["ready"]:
                logger.warning(f"[trace_id={trace_id}] [chat] final_status=empty_index reason=知识库未准备完成")
                response.status_code = 503

                metrics["status"] = "empty_index"
                metrics["action"] = action or ""
                metrics["reason"] = "知识库未准备完成"

                bad_case["route"] = action or "rag"
                bad_case["error_type"] = "index_not_ready"
                bad_case["note"] = "知识库未准备完成，请检查知识库"
                write_bad_cases(bad_case)

                session_should_store = True
                session_assistant_content = rag_status["message"] or "知识库未准备完成"

                return build_chat_response(
                    code="NO_DOCUMENTS",
                    message=rag_status["message"],
                    trace_id=trace_id,
                    question=original_question,
                    need_clarification=False,
                    status="empty_index",
                    clarification_question="",
                    reason="知识库未准备完成",
                    tool_trace=[rewrite_trace]
                )

        try:
            agent_start = time.perf_counter()

            result = run_agent_v2(
                write_bad_cases=write_bad_cases,
                question=effective_question,
                top_k=final_top_k,
                decompose_top_k=decompose_top_k,
                decision=decision,
                trace_id=trace_id,
                session_id=session_id or None,
                bad_case=bad_case,
            )
            result = attach_query_rewrite_trace(result, rewrite_trace)

            agent_cost_ms = int((time.perf_counter() - agent_start) * 1000)
            metrics["agent_cost_ms"] = agent_cost_ms

            logger.info(
                f"[trace_id={trace_id}] [agent] run_agent_v2_end "
                f"status={result.get('status', '')} "
                f"agent_cost_ms={agent_cost_ms} "
                f"answer_len={len(result.get('answer', '') or '')} "
                f"reference_count={len(result.get('references') or [])}"
            )

            trace_metrics = collect_trace_metrics(result)

            metrics["status"] = normalize_metrics_status(result.get("status", ""))
            metrics["reference_count"] = len(result.get("references") or [])
            metrics["retrieved_count"] = trace_metrics["retrieved_count"]
            metrics["tool_called"] = trace_metrics["tool_called"]
            metrics["tool_success"] = trace_metrics["tool_success"]
            metrics["tool_name"] = trace_metrics["tool_name"]
            metrics["router_llm_used"] = trace_metrics["router_llm_used"]
            metrics["decompose_llm_used"] = trace_metrics["decompose_llm_used"]
            metrics["answer_llm_used"] = trace_metrics["answer_llm_used"]
            metrics["llm_call_count"] = trace_metrics["llm_call_count"]
            for timing_key in [
                "retrieve_cost_ms",
                "decompose_cost_ms",
                "rerank_cost_ms",
                "retrieve_with_sub_queries_cost_ms",
                "context_build_cost_ms",
                "llm_cost_ms",
                "parse_cost_ms",
                "rag_answer_cost_ms",
            ]:
                metrics[timing_key] = int((trace_metrics["timings"].get(timing_key, 0) or 0))
            metrics["action"] = result.get("action", action or "")
            metrics["reason"] = result.get("reason", "")

        except Exception as e:
            logger.exception(f"[trace_id={trace_id}] [answer] 处理失败 error={e}")

            metrics["status"] = "internal_error"
            metrics["action"] = action or ""
            metrics["tool_name"] = decision.get("tool_name", "")
            metrics["tool_called"] = bool(decision.get("tool_name", ""))
            metrics["tool_success"] = False
            metrics["router_llm_used"] = bool(
                decision.get(
                    "router_llm_used",
                    decision.get("router_source") in {"llm", "rule_based_fallback"}
                )
            )
            metrics["llm_call_count"] = (
                int(metrics["query_rewrite_llm_used"])
                + int(metrics["router_llm_used"])
            )
            metrics["reason"] = str(e)

            bad_case["route"] = action or decision.get("decision", "")
            bad_case["error_type"] = "internal_error"
            bad_case["note"] = f"Agent Runtime 执行异常：{str(e)}"
            write_bad_cases(bad_case)

            response.status_code = 500
            return build_chat_response(
                code="MODEL_ERROR",
                message="处理失败，请稍后重试",
                trace_id=trace_id,
                question=original_question,
                need_clarification=False,
                status="error",
                clarification_question="",
                reason="工具调用或模型调用异常",
                tool_trace=[rewrite_trace]
            )

        response_config = CHAT_RESULT_RESPONSE_MAP.get(result["status"])
        if response_config:
            logger.warning(
                f"[trace_id={trace_id}] [chat] "
                f"final_status={response_config['log_status']} "
                f"reason={result.get('reason', response_config['default_reason'])}"
            )
            response.status_code = 200
            session_should_store = True
            session_assistant_content = (
                response_config.get("answer", result.get("answer", ""))
                or result.get("clarification_question", "")
                or result.get("reason", response_config["default_reason"])
            )
            return build_chat_response_from_result(
                result=result,
                trace_id=trace_id,
                question=original_question,
                response_config=response_config,
            )

        logger.info(
            f"[trace_id={trace_id}] [chat] final_status=ok reason={result.get('reason', '正常回答')}"
        )
        response.status_code = 200
        session_should_store = True
        session_assistant_content = result.get("answer", "") or result.get("reason", "正常回答")
        return build_chat_response(
            code=0,
            message="success",
            trace_id=trace_id,
            question=original_question,
            answer=result["answer"],
            references=result.get("references", []),
            retrieval_confidence=result.get("retrieval_confidence", 0.0),
            answer_confidence=result.get("answer_confidence", 0.0),
            need_clarification=result.get("need_clarification", False),
            status="ok",
            clarification_question=result.get("clarification_question", ""),
            reason=result.get("reason", "正常回答"),
            action=result.get("action", ""),
            tool_result_summary=result.get("tool_result_summary", ""),
            tool_trace=result.get("tool_trace", []),
            router_llm_used=result.get("router_llm_used"),
            decompose_llm_used=result.get("decompose_llm_used"),
            answer_llm_used=result.get("answer_llm_used"),
            llm_call_count=result.get("llm_call_count"),
        )

    except HTTPException:
        metrics["status"] = "internal_error"
        metrics["reason"] = "HTTPException"

        bad_case["error_type"] = "internal_error"
        bad_case["note"] = "HTTPException"
        write_bad_cases(bad_case)
        raise
    except Exception as e:
        logger.exception(f"[trace_id={trace_id}] [chat] 未知异常 error={e}")

        metrics["status"] = "internal_error"
        metrics["reason"] = str(e)

        response.status_code = 500
        return build_chat_response(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            trace_id=trace_id,
            question=original_question,
            need_clarification=False
        )


    finally:
        process_end_perf = time.perf_counter()

        metrics["end_time"] = now_iso()

        # /chat 内部真正处理耗时
        metrics["latency_ms"] = int(
            (process_end_perf - process_start_perf) * 1000
        )

        # 从 HTTP 请求进入，到 /chat 处理结束的总耗时
        metrics["total_cost_ms"] = int(
            (process_end_perf - request_enter_perf) * 1000
        )

        if session_should_store:
            save_session_turn(
                session_id=session_id,
                user_question=original_question,
                assistant_content=session_assistant_content,
                trace_id=trace_id,
            )

        logger.info(
            f"[trace_id={trace_id}] [chat] process_end "
            f"status={metrics.get('status')} "
            f"queue_wait_ms={metrics.get('queue_wait_ms')} "
            f"latency_ms={metrics.get('latency_ms')} "
            f"total_cost_ms={metrics.get('total_cost_ms')} "
            f"route_cost_ms={metrics.get('route_cost_ms')} "
            f"rag_ready_check_cost_ms={metrics.get('rag_ready_check_cost_ms')} "
            f"agent_cost_ms={metrics.get('agent_cost_ms')} "
            f"retrieve_cost_ms={metrics.get('retrieve_cost_ms')} "
            f"decompose_cost_ms={metrics.get('decompose_cost_ms')} "
            f"rerank_cost_ms={metrics.get('rerank_cost_ms')} "
            f"llm_cost_ms={metrics.get('llm_cost_ms')} "
            f"llm_call_count={metrics.get('llm_call_count')} "
            f"query_rewrite_used={metrics.get('query_rewrite_used')} "
            f"query_rewrite_llm_used={metrics.get('query_rewrite_llm_used')} "
            f"effective_question_len={metrics.get('effective_question_len')}"
        )

        log_and_write_metrics(metrics)

_diagnostics_router, _diagnostics_handlers = create_diagnostics_router(
    DiagnosticsDependencies(
        settings=Settings,
        chat_request=ChatRequest,
        search_result=SearchResult,
        search_data=SearchData,
        search_response=SearchResponse,
        preview_docs=lambda: preview_docs(),
        preview_chunks=lambda *args, **kwargs: preview_chunks(*args, **kwargs),
        get_rag_status=lambda: get_rag_status(),
        search_relevant_chunks=lambda *args, **kwargs: search_relevant_chunks(
            *args, **kwargs
        ),
        rerank=lambda *args, **kwargs: rerank(*args, **kwargs),
        should_refuse=lambda *args, **kwargs: should_refuse(*args, **kwargs),
        safe_float=lambda value: safe_float(value),
        route_agent_action=lambda *args, **kwargs: route_agent_action(
            *args, **kwargs
        ),
        read_recent_metrics=lambda limit: read_recent_metrics(limit),
    )
)
app.include_router(_diagnostics_router)
docs_preview = _diagnostics_handlers["docs_preview"]
chunks_preview = _diagnostics_handlers["chunks_preview"]
search_api = _diagnostics_handlers["search_api"]
kb_summary = _diagnostics_handlers["kb_summary"]
decide_api = _diagnostics_handlers["decide_api"]
metrics_summary = _diagnostics_handlers["metrics_summary"]
