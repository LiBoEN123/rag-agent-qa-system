from fastapi import FastAPI, HTTPException, Query, Response, Request
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, TypeAlias
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from app.rag import (
    preview_docs,
    preview_chunks,
    search_relevant_chunks,
    rerank,
    get_rag_status,
    init_rag,
    needs_rebuild,
    load_index_cache,
    index_cache_exists,
    rebuild_and_persist_index,
)
from app.config import Settings
from app.untils import get_logger
from app.agent import route_agent_action, run_agent_once
from app.observability import create_trace_id

PublicStatus: TypeAlias = Literal[
    "ok",
    "no_evidence",
    "empty_index",
    "error",
    "clarify",
    "out_of_scope",
]
app = FastAPI(title='Mini Rag API', version='1.0.0')
logger = get_logger('chat_service')

METRICS_PATH = Path("logs") / "metrics.jsonl"
BAD_CASES_PATH = Path("logs") / "bad_cases.jsonl"


# =========================
# API 数据模型
# =========================

class ChatRequest(BaseModel):
    question:str = Field(..., description='用户问题')
    top_k:int | None=Field(None, ge=1, le=10, description='召回条数')

    model_config = ConfigDict(
        json_schema_extra={
            "example":{
              "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
              "top_k": 5
            }
        }
    )

class ChatReferences(BaseModel):
    reference_id:str
    file_name:str
    source_path:str
    chunk_id:str | int
    position: int | None = None
    snippet: str
    score: float
    rerank_score: float

class ChatData(BaseModel):
    question:str
    answer:str
    references:list[ChatReferences]=Field(default_factory=list)
    confidence:float = Field(..., ge=0.0, le=1.0, description="回答置信度，0~1")
    need_clarification:bool = Field(..., description="是否建议用户补充问题")
    status: PublicStatus
    clarification_question:str = Field(..., description="补充提示")
    reason:str = Field(..., description="理由")
    action:str = Field(default="", description="本次Agent选择的动作")
    tool_result_summary:str = Field(default="", description="工具结果摘要")
    tool_trace:list[dict] = Field(default_factory=list, description="工具调用或路径追踪")

class ChatResponse(BaseModel):
    code:int|str
    message:str
    trace_id:str = Field(default="", description="本次请求追踪的id")
    data:ChatData

    model_config = ConfigDict(
        json_schema_extra={
            "example":{
              "code": 0,
              "message": "success",
              "data": {
                "answer": "每个 Vue 应用都是通过 `createApp` 函数创建一个新的应用实例。",
                "references": [
                  {
                    "reference_id": "Reference 1",
                    "file_name": "application.md",
                    "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
                    "chunk_id": 1,
                    "position": 1,
                    "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用实例{#the-application-instance}每个Vue应用都是通过[`createApp`]",
                    "score": 1
                  }
                ],
                "confidence": 0.519,
                "need_clarification": False,
                "answer_status": "answered",
                "clarification_question": "",
                "reason": "检索命中有效证据，且答案成功绑定引用"
              }
            }
        }
    )

class SearchResult(BaseModel):
    reference_id:str
    source_path:str
    file_name:str
    chunk_id:str | int
    position:int | None=None
    snippet: str
    score:float
    bm25_score:float
    vector_score:float
    rerank_score:float

class SearchData(BaseModel):
    question:str
    status:PublicStatus
    reason:str=""
    results:list[SearchResult]=Field(default_factory=list)

class SearchResponse(BaseModel):
    code:int | str
    message:str
    data:SearchData

    model_config = ConfigDict(
        json_schema_extra={
            "example":{
                  "code": 0,
                  "message": "success",
                  "data": {
                    "question": "每个 Vue 应用都是通过什么函数创建应用实例的？",
                    "references": [
                      {
                        "reference_id": "Reference 1",
                        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
                        "file_name": "application.md",
                        "chunk_id": 1,
                        "position": 1,
                        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用实例{#the-application-instance}每个Vue应用都是通过[`createApp`]",
                        "score": 1,
                        "bm25_score": 34.055355137731354,
                        "vector_score": 0.741374135017395
                      },
                      {
                        "reference_id": "Reference 2",
                        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
                        "file_name": "application.md",
                        "chunk_id": 10,
                        "position": 10,
                        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>多个应用实例{#multiple-application-instances}应用实例并不只限于一个。`cre",
                        "score": 0.9164628982543945,
                        "bm25_score": 30.0893694535201,
                        "vector_score": 0.7052635550498962
                      },
                      {
                        "reference_id": "Reference 3",
                        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
                        "file_name": "application.md",
                        "chunk_id": 9,
                        "position": 9,
                        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用配置{#app-configurations}app.component('TodoDeleteButto",
                        "score": 0.7920963764190674,
                        "bm25_score": 26.642277059679156,
                        "vector_score": 0.6124808192253113
                      },
                      {
                        "reference_id": "Reference 4",
                        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\application.md",
                        "file_name": "application.md",
                        "chunk_id": 8,
                        "position": 8,
                        "snippet": "[标题路径]创建一个Vue应用{#creating-a-vue-application}>应用配置{#app-configurations}应用实例会暴露一个`.config`对象允许我们配置一些应用",
                        "score": 0.7547773122787476,
                        "bm25_score": 22.68307872610552,
                        "vector_score": 0.6310862302780151
                      },
                      {
                        "reference_id": "Reference 5",
                        "source_path": "D:\\python-learn\\mini-rag\\data\\kb\\introduction.md",
                        "file_name": "introduction.md",
                        "chunk_id": 198,
                        "position": 6,
                        "snippet": "[标题路径]简介{#introduction}>什么是Vue？{#what-is-vue}Vue(发音为/vjuː/，类似**view**)是一款用于构建用户界面的JavaScript框架。它基于标准",
                        "score": 0.7235828638076782,
                        "bm25_score": 18.788527818590843,
                        "vector_score": 0.6559301614761353
                      }
                    ]
                  }
                }
        }
    )



# =========================
# bad case 分类
# =========================
# 真正应该进入 bad_cases.jsonl 的错误类型

REAL_BAD_CASE = {
    "tool_error",            # 工具执行异常、工具返回格式异常
    "parse_error",           # 模型没有按 JSON 格式返回
    "reference_error",       # 答案和引用不匹配，或答案有内容但引用为空
    "model_error",           # 模型调用异常
    "internal_error",        # 后端内部异常
    "agent_decision_error",  # Agent 决策出现未知 action
    "index_not_ready",       # 知识库未准备完成
}

# 这些属于正常业务结果，不应该进入 bad_cases.jsonl

EXPECTED_NON_BAD_CASE_TYPES = {
    "",
    "empty_question",
    "invalid_input",
    "clarify_needed",
    "ambiguous_error",
    "expected_refusal",
    "out_of_scope",
    "out_of_scope_error",
    "out_of_scope_refused",
    "no_relevant_info",
    "insufficient",
    "retrieval_error",   # 注意：这里先不把“没检索到”当成 bad case
}

def should_write_bad_case(record: dict) -> bool:
    # 判断是否真的要写入 bad_cases.jsonl。
    error_type = str(record.get("error_type", "") or "").strip()

    if error_type in REAL_BAD_CASE:
        return True

    if error_type in EXPECTED_NON_BAD_CASE_TYPES:
        return False

    # 防御：如果以后出现了未知 error_type，先不要静默写入。
    logger.info(f"[trace_id={record.get('trace_id', '')}] [bad_case] skip unknown error_type={error_type}")

    return False

def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")

@app.middleware("http")
async def trace_and_timing_middleware(request: Request, call_next):
    if request.url.path != "/chat":
        return await call_next(request)

    trace_id = request.headers.get("X-Trace-Id") or create_trace_id()

    request_enter_perf = time.perf_counter()
    request_enter_time = now_iso()

    request.state.trace_id = trace_id
    request.state.request_enter_perf = request_enter_perf
    request.state.request_enter_time = request_enter_time

    logger.info(
        f"[trace_id={trace_id}] [http] request_enter "
        f"path={request.url.path} "
        f"method={request.method} "
        f"request_enter_time={request.state.request_enter_time}"
    )

    try:
        response = await call_next(request)
    except Exception as e:
        total_cost_ms = int((time.perf_counter() - request_enter_perf) * 1000)
        logger.info(
            f"[trace_id={trace_id}] [http] request_error "
            f"total_cost_ms={total_cost_ms} "
            f"error={repr(e)}"
        )
        raise

    total_cost_ms = int((time.perf_counter() - request_enter_perf) * 1000)

    logger.info(
        f"[trace_id={trace_id}] [http] response_end "
        f"status_code={response.status_code} "
        f"total_cost_ms={total_cost_ms} "
    )

    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Total-Cost-Ms"] = str(total_cost_ms)

    return response


def normalize_metrics_status(result_status: str) -> str:

    status_map = {
        "OK": "success",
        "NO_RELEVANT_INFO": "no_relevant_info",
        "TOOL_ERROR": "tool_error",
        "MODEL_BAD_OUTPUT": "model_bad_output",
        "NEED_CLARIFICATION": "clarify",
        "OUT_OF_SCOPE": "out_of_scope",
        "MODEL_ERROR": "model_error",
    }

    return status_map.get(result_status, "internal_error")

def collect_trace_metrics(result: dict) -> dict:

    tool_trace = result.get("tool_trace") or []

    retrieved_count = 0
    tool_called = False
    tool_success = False
    tool_name = ""

    real_tools = {"simple_calculator", "list_documents", "get_chunk_text"}

    for step in tool_trace:
        current_tool_name = step.get("tool_name", "")

        if current_tool_name == "search_knowledge_base":
            retrieved_count = step.get("final_chunk_count", retrieved_count)

        if current_tool_name in real_tools:
            tool_name = current_tool_name
            tool_called = True

            if step.get("tool_status") == "success":
                tool_success = True

    return {
        "retrieved_count": retrieved_count,
        "tool_called": tool_called,
        "tool_success": tool_success,
        "tool_name": tool_name,
        "timings": result.get("timings") or {},
    }

def write_metrics(record: dict) -> None:

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_bad_cases(record: dict) -> None:
    """
       写入真正失败案例。

       注意：
       这个函数内部会先判断 should_write_bad_case(record)。
       所以即使 agent.py 误调用了 write_bad_cases，
       空问题、正常拒答、正常澄清、正常无证据也不会写进 bad_cases.jsonl。
       """

    if not should_write_bad_case(record):
        logger.info(
            f"[trace_id={record.get('trace_id', '')}] [bad_case] skip "
            f"error_type={record.get('error_type', '')} "
            f"note={record.get('note', '')}"
        )
        return

    BAD_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)

    normalized_record = {
        "created_at": now_iso(),
        "trace_id": record.get("trace_id", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "references": record.get("references", []),
        "route": record.get("route", ""),
        "error_type": record.get("error_type", ""),
        "note": record.get("note", ""),
    }

    with BAD_CASES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(normalized_record, ensure_ascii=False) + "\n")

    logger.info(
        f"[trace_id={record.get('trace_id', '')}] [bad_case] written "
        f"route={record.get('route', '')} "
        f"error_type={record.get('error_type', '')} "
        f"note={record.get('note', '')}"
    )

def log_and_write_metrics(record: dict) -> None:

    logger.info(
        f"[trace_id={record.get('trace_id')}] "
        "[metrics] "
        f"queue_wait_ms={record.get('queue_wait_ms')} "
        f"latency_ms={record.get('latency_ms')} "
        f"total_cost_ms={record.get('total_cost_ms')} "
        f"route_cost_ms={record.get('route_cost_ms')} "
        f"rag_ready_check_cost_ms={record.get('rag_ready_check_cost_ms')} "
        f"agent_cost_ms={record.get('agent_cost_ms')} "
        f"retrieve_cost_ms={record.get('retrieve_cost_ms')} "
        f"decompose_cost_ms={record.get('decompose_cost_ms')} "
        f"rerank_cost_ms={record.get('rerank_cost_ms')} "
        f"retrieve_with_sub_queries_cost_ms={record.get('retrieve_with_sub_queries_cost_ms')} "
        f"context_build_cost_ms={record.get('context_build_cost_ms')} "
        f"llm_cost_ms={record.get('llm_cost_ms')} "
        f"parse_cost_ms={record.get('parse_cost_ms')} "
        f"rag_answer_cost_ms={record.get('rag_answer_cost_ms')} "
        f"status={record.get('status')} "
        f"reference_count={record.get('reference_count')} "
        f"retrieved_count={record.get('retrieved_count')} "
        f"tool_called={record.get('tool_called')} "
        f"tool_success={record.get('tool_success')} "
        f"tool_name={record.get('tool_name')} "
        f"action={record.get('action')} "
        f"reason={record.get('reason')}"
    )

    write_metrics(record)

# =========================
# 响应构造与状态转换
# =========================
def build_chat_response(code:int|str,
                        message:str,
                        trace_id:str,
                        question:str,
                        answer:str="",
                        references:list | None = None,
                        confidence: float = 0.0,
                        need_clarification: bool = False,
                        status: PublicStatus = "ok",
                        clarification_question:str = "",
                        reason:str = "",
                        action:str = "",
                        tool_result_summary:str = "",
                        tool_trace:list | None = None,
                        ) -> ChatResponse:
    return ChatResponse(code=code,
                        message=message,
                        trace_id=trace_id,
                        data=ChatData(
                            question=question,
                            answer=answer,
                            references=references or [],
                            confidence=confidence,
                            need_clarification=need_clarification,
                            status=status,
                            clarification_question=clarification_question,
                            reason=reason,
                            action=action,
                            tool_result_summary=tool_result_summary,
                            tool_trace=tool_trace or []
                            )
                        )

def get_normalize_status(result: dict) -> str:
    status = result.get("status", "")

    status_map = {
        "OK": "ok",
        "NO_RELEVANT_INFO": "no_evidence",
        "NEED_CLARIFICATION": "clarify",
        "TOOL_ERROR": "error",
        "OUT_OF_SCOPE": "out_of_scope",
        "MODEL_BAD_OUTPUT": "error",
    }

    return status_map.get(status, "unknown")

def log_final_status(trace_id: str, result: dict):
    answer_status = get_normalize_status(result)
    reference_count = len(result.get("references") or [])
    need_clarification = result.get("need_clarification", False)

    logger.info(
        f"[trace_id={trace_id}] [final] answer_status={answer_status} "
        f"reference_count={reference_count} need_clarification={need_clarification}"
    )

def read_recent_metrics(limit: int = 20) -> list[dict]:

    if not METRICS_PATH.exists():
        return []

    lines = METRICS_PATH.read_text(encoding='utf-8').splitlines()
    recent_lines = lines[-limit:]

    records = []

    for line in recent_lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records



# =========================
# FastAPI 路由
# =========================

@app.on_event("startup")
def start_on_event():
    Settings.validate()

    data_dir = str(Settings.RAG_FILE_PATH)
    need, reason = needs_rebuild(data_dir)
    logger.info(f"[startup] needs_rebuild={need}, reason={reason}")

    should_force_rebuild = (
        need
        or not index_cache_exists()
        or Settings.AUTO_REBUILD_ON_STARTUP
    )

    if should_force_rebuild:
        ok, msg = rebuild_and_persist_index(
            data_dir=data_dir,
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP
        )
        logger.info(f"[startup] {msg}")

        if not ok:
            logger.warning("[startup] 重建失败，RAG 可能不可用")

    else:
        loaded, msg = load_index_cache()
        logger.info(f"[startup] {msg}")

        if not loaded:
            logger.warning("[startup] 缓存加载失败，降级为重建")
            ok, msg = rebuild_and_persist_index(
                data_dir=data_dir,
                chunk_size=Settings.CHUNK_SIZE,
                chunk_overlap=Settings.CHUNK_OVERLAP
            )
            logger.info(f"[startup] {msg}")

@app.get("/health")
def health():
    return {
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok"
  }
}

@app.post("/chat", response_model=ChatResponse)
def ask_api(req: ChatRequest, request: Request, response: Response):
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

    question = (req.question or "").strip()

    logger.info(
        f"[trace_id={trace_id}] [chat] process_start "
        f"queue_wait_ms={queue_wait_ms} "
        f"question_len={len(question)} "
        f"preview={question[:50]}"
    )

    bad_case = {
        "created_at": now_iso(),
        "trace_id":trace_id,
        "question":question,
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
        "question_len": len(question),
        "reference_count": 0,
        "retrieved_count": 0,
        "tool_called": False,
        "tool_success": False,
        "tool_name": "",
        "action": "",
        "reason": "",
    }

    try:
        if not question:
            logger.warning(f"[trace_id={trace_id}] [chat] final_status=clarify reason=用户未输入问题")
            response.status_code = 400

            metrics["status"] = "clarify"
            metrics["reason"] = "用户未输入问题"

            return build_chat_response(
                code="EMPTY_QUESTION",
                message="问题不能为空，请输入你想查询的内容",
                trace_id=trace_id,
                question=question,
                confidence=0.0,
                need_clarification=True,
                status="clarify",
                clarification_question="请直接输入你想查询的具体问题，例如：年假申请流程是什么？",
                reason="用户未输入问题"
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K

        route_start = time.perf_counter()

        decision = route_agent_action(question, final_top_k)

        route_cost_ms = int((time.perf_counter() - route_start) * 1000)
        metrics["route_cost_ms"] = route_cost_ms

        action = decision.get("action")

        logger.info(
            f"[trace_id={trace_id}] [agent] route_end "
            f"action={action} "
            f"route_cost_ms={route_cost_ms} "
            f"reason={decision.get('reason', '')}"
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

                return build_chat_response(
                    code="NO_DOCUMENTS",
                    message=rag_status["message"],
                    trace_id=trace_id,
                    question=question,
                    confidence=0.0,
                    need_clarification=False,
                    status="empty_index",
                    clarification_question="",
                    reason="知识库未准备完成"
                )

        try:
            agent_start = time.perf_counter()

            result = run_agent_once(
                write_bad_cases,
                question,
                final_top_k,
                decision,
                trace_id,
                bad_case
            )

            agent_cost_ms = int((time.perf_counter() - agent_start) * 1000)
            metrics["agent_cost_ms"] = agent_cost_ms

            logger.info(
                f"[trace_id={trace_id}] [agent] run_agent_once_end "
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
            metrics["reason"] = str(e)

            bad_case["route"] = action or decision.get("decision", "")
            bad_case["error_type"] = "internal_error"
            bad_case["note"] = f"run_agent_once 执行异常：{str(e)}"
            write_bad_cases(bad_case)

            response.status_code = 500
            return build_chat_response(
                code="MODEL_ERROR",
                message="处理失败，请稍后重试",
                trace_id=trace_id,
                question=question,
                confidence=0.0,
                need_clarification=False,
                status="error",
                clarification_question="",
                reason="工具调用或模型调用异常"
            )

        if result["status"] == "MODEL_ERROR":
            response.status_code = 200
            return build_chat_response(
                code="MODEL_ERROR",
                message="模型调用失败",
                trace_id=trace_id,
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=0.0,
                need_clarification=False,
                status="error",
                clarification_question="",
                reason=result.get("reason", "模型调用失败"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        if result["status"] == "NEED_CLARIFICATION":
            logger.warning(
                f"[trace_id={trace_id}] [chat] final_status=need_clarification reason=问题信息不足"
            )
            response.status_code = 200
            return build_chat_response(
                code="NEED_CLARIFICATION",
                message="问题信息不足，需要补充后再回答",
                trace_id=trace_id,
                question=question,
                answer="",
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=True,
                status="clarify",
                clarification_question=result.get("clarification_question", ""),
                reason=result.get("reason", "问题缺少关键限定信息"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        if result["status"] == "NO_RELEVANT_INFO":
            logger.warning(
                f"[trace_id={trace_id}] [chat] final_status=no_evidence reason={result.get('reason', '检索证据不足')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="NO_RELEVANT_INFO",
                message="我没有在文档中找到相关信息",
                trace_id=trace_id,
                question=question,
                answer=result.get("answer", ""),
                references=result.get("references", []),
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                status="no_evidence",
                clarification_question="",
                reason=result.get("reason", "模型判断当前上下文不足以支撑回答"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        if result["status"] == "TOOL_ERROR":
            logger.warning(
                f"[trace_id={trace_id}] [chat] final_status=error reason={result.get('reason', '工具调用失败')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="TOOL_ERROR",
                message="工具调用失败",
                trace_id=trace_id,
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                status="error",
                clarification_question=result.get("clarification_question", ""),
                reason=result.get("reason", "工具调用失败"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        if result["status"] == "OUT_OF_SCOPE":
            logger.warning(
                f"[trace_id={trace_id}] [chat] final_status=out_of_scope reason={result.get('reason', '超出知识库范围')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="OUT_OF_SCOPE",
                message="问题超出当前知识库范围",
                trace_id=trace_id,
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=False,
                status="out_of_scope",
                clarification_question="",
                reason=result.get("reason", "问题超出知识库范围"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        if result["status"] == "MODEL_BAD_OUTPUT":
            logger.warning(
                f"[trace_id={trace_id}] [chat] final_status=error reason={result.get('reason', '模型输出格式异常')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="MODEL_BAD_OUTPUT",
                message="模型输出格式异常，请稍候重试",
                trace_id=trace_id,
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                status="error",
                clarification_question="",
                reason=result.get("reason", "模型返回结构不符合预期"),
                action=result.get("action", ""),
                tool_result_summary=result.get("tool_result_summary", ""),
                tool_trace=result.get("tool_trace", [])
            )

        logger.info(
            f"[trace_id={trace_id}] [chat] final_status=ok reason={result.get('reason', '正常回答')}"
        )
        response.status_code = 200
        return build_chat_response(
            code=0,
            message="success",
            trace_id=trace_id,
            question=question,
            answer=result["answer"],
            references=result.get("references", []),
            confidence=result.get("confidence", 0.0),
            need_clarification=result.get("need_clarification", False),
            status="ok",
            clarification_question=result.get("clarification_question", ""),
            reason=result.get("reason", "正常回答"),
            action=result.get("action", ""),
            tool_result_summary=result.get("tool_result_summary", ""),
            tool_trace=result.get("tool_trace", [])
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
            question=question,
            confidence=0.0,
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
            f"llm_cost_ms={metrics.get('llm_cost_ms')}"
        )

        log_and_write_metrics(metrics)

@app.get("/docs-preview")
def docs_preview():
    docs = preview_docs()
    return {"count":len(docs), "data":docs}

@app.get("/chunks-preview")
def chunks_preview(
        max_len:int = Query(default=300, ge=1, le=1000),
        limit:int = Query(70, ge=1, le=1000),
        file_name:str | None=Query(None)):
    data = preview_chunks(max_len,limit, file_name)
    return {"count":len(data), "data":data}

@app.post("/search", response_model=SearchResponse)
def search_api(req:ChatRequest, response:Response):
    question = (req.question or "").strip()
    logger.info(f"[search] 收到请求，question_len={len(question)}，question_preview={question[:50]}")

    try:
        if not question:
            logger.warning(f"[search] 空问题请求")
            response.status_code = 400
            return SearchResponse(
                code="EMPTY_QUESTION",
                message="问题不能为空",
                data=SearchData(
                    question=question,
                    status="error",
                    reason="用户未输入问题",
                    results=[]
                )
            )

        rag_status = get_rag_status()
        if not rag_status["ready"]:
            logger.info("[search] final_status=empty_index reason=知识库未准备完成")
            response.status_code = 503
            return SearchResponse(
                code="EMPTY_INDEX",
                message=rag_status["message"],
                data=SearchData(
                    question=question,
                    status="empty_index",
                    reason="知识库未准备完成",
                    results=[]
                )
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K
        retrieved_chunks = search_relevant_chunks(question, final_top_k)

        if not retrieved_chunks:
            logger.info("[search] final_status=no_evidence reason=没有检索到结果")
            response.status_code = 200
            return SearchResponse(
                code=0,
                message="success",
                data=SearchData(
                    question=question,
                    status="no_evidence",
                    reason="未检索到结果",
                    results=[]
                )
            )

        reranked_chunks = rerank(retrieved_chunks, question)
        final_chunks = reranked_chunks[:final_top_k]

        if not final_chunks:
            logger.info("[search] final_status=no_evidence reason=重排后没有可用结果")
            response.status_code = 200
            return SearchResponse(
                code=0,
                message="success",
                data=SearchData(
                    question=question,
                    status="no_evidence",
                    reason="重排后没有可用结果",
                    results=[]
                )
            )

        top_vector_score = max(
            float(item.get("vector_score", 0.0))
            for item in final_chunks
        )

        if top_vector_score < Settings.VECTOR_THRESHOLD:
            logger.warning(f"[search] vector 分数过低 top_vector_score={top_vector_score}")
            response.status_code = 200
            return SearchResponse(
                code=0,
                message="success",
                data=SearchData(
                    question=question,
                    status="no_evidence",
                    reason=f"vector 分数过低：{top_vector_score:.4f}",
                    results=[]
                )
            )

        for rank, item in enumerate(final_chunks, start=1):
            item["reference_id"] = f"Reference {rank}"

        results = [
            SearchResult(
                reference_id=item["reference_id"],
                source_path=item["source_path"],
                file_name=item["file_name"],
                chunk_id=item["chunk_id"],
                position=item.get("position"),
                snippet=item["snippet"],
                score=item["hybrid_score"],
                bm25_score=item["bm25_score"],
                vector_score=item["vector_score"],
                rerank_score=item.get("rerank_score", 0.0)
            )
            for item in final_chunks
        ]

        score_brief = [
            {
                "chunk_id": item["chunk_id"],
                "score": round(item["hybrid_score"], 4),
                "bm25": round(item["bm25_score"], 4),
                "vector": round(item["vector_score"], 4),
                "rerank": round(item.get("rerank_score", 0.0), 4),
            }
            for item in final_chunks
        ]
        logger.info(f"[search] retrieve_hits={score_brief}")

        logger.info("[search] final_status=ok reason=检索成功")
        response.status_code = 200
        return SearchResponse(
            code=0,
            message="success",
            data=SearchData(
                question=question,
                status="ok",
                reason="检索成功",
                results=results
            )
        )

    except Exception as e:
        logger.exception(f"[search] final_status=error reason=search 接口处理异常 error={e}")
        response.status_code = 500
        return SearchResponse(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            data=SearchData(
                question=question,
                status="error",
                reason="search 接口处理异常",
                results=[]
            )
        )

@app.post("/kb_summary")
def kb_summary():
    status = get_rag_status()
    docs = preview_docs()

    return {
        "code":0,
        "message":"success",
        "data":{
            "ready":status.get("ready"),
            "message":status.get("message"),
            "doc_count":status.get("doc_count"),
            "chunk_count":status.get("chunk_count"),
            "file_names":[doc["file_name"] for doc in docs],
            "documents":docs
        }
    }

@app.post("/decide")
def decide_api(req:ChatRequest, response:Response):
    question = (req.question or "").strip()

    if not question:
        response.status_code = 400
        return {
            "code": "EMPTY_QUESTION",
            "message": "问题不能为空",
            "data": {
                "question": question,
                "decision": "clarify",
                "reason": "用户未输入问题"
            }
        }

    final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K

    decision = route_agent_action(question, final_top_k)

    logger.info(f"[decision] question={question[:50]} decision={decision}")

    return {
        "code": 0,
        "message": "success",
        "data": {
            "question": question,
            "decision": decision,
        }
    }

@app.get("/metrics-summary")
def metrics_summary(limit: int = Query(20, ge=1, le=200)):
    records = read_recent_metrics(limit)

    status_counter = Counter(record.get("status", "unknown") for record in records)

    if records:
        avg_latency_ms = round(
            sum(record.get("latency_ms", 0) for record in records) /
            len(records), 2)

    else:
        avg_latency_ms = 0

    return {
        "code": 0,
        "message": "success",
        "data": {
            "limit": limit,
            "total": len(records),
            "status_count": dict(status_counter),
            "success_count": status_counter.get("success", 0),
            "no_relevant_info_count": status_counter.get("no_relevant_info", 0),
            "tool_error_count": status_counter.get("tool_error", 0),
            "model_bad_output_count": status_counter.get("model_bad_output", 0),
            "model_error_count": status_counter.get("model_error", 0),
            "clarify_count": status_counter.get("clarify", 0),
            "out_of_scope_count": status_counter.get("out_of_scope", 0),
            "empty_index_count": status_counter.get("empty_index", 0),
            "internal_error_count": status_counter.get("internal_error", 0),
            "avg_latency_ms": avg_latency_ms,
        }
    }
