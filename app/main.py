from fastapi import FastAPI, HTTPException, Query, Response
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
    get_rag_status,
    needs_rebuild,
    load_index_cache,
    index_cache_exists,
    rebuild_and_persist_index,
)
from app.config import Settings
from app.untils import get_logger
from app.extract_refund_policy import extract_refund_policy
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

def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")

def normalize_metrics_status(result_status: str) -> str:

    status_map = {
        "OK": "success",
        "NO_RELEVANT_INFO": "no_relevant_info",
        "TOOL_ERROR": "tool_error",
        "MODEL_BAD_OUTPUT": "model_bad_output",
        "NEED_CLARIFICATION": "no_relevant_info",
        "OUT_OF_SCOPE": "no_relevant_info",
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
            retrieved_count = step.get("retrieved_count", retrieved_count)

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
    }

def write_metrics(record: dict) -> None:

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_and_write_metrics(record: dict) -> None:

    logger.info(
        f"[trace_id={record.get('trace_id')}] "
        "[metrics] "
        f"latency_ms={record.get('latency_ms')} "
        f"status={record.get('status')} "
        f"reference_count={record.get('reference_count')} "
        f"retrieved_count={record.get('retrieved_count')} "
        f"tool_called={record.get('tool_called')} "
        f"tool_success={record.get('tool_success')} "
        f"tool_name={record.get('tool_name')}"
    )

    write_metrics(record)


# =========================
# 接口变量定义
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

class ExtractRequest(BaseModel):
    text: str = Field(..., description="待抽取文本")

class ExtractData(BaseModel):
    is_refundable: bool | None = None
    refund_window_days: int | None = None
    exceptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

class ExtractResponse(BaseModel):
    code: int | str
    message: str
    data: ExtractData

# chat接口回复结构定义
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
# 可调用接口
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
def ask_api(req: ChatRequest, response: Response):
    start_perf = time.perf_counter()
    start_time = now_iso()

    trace_id = create_trace_id()
    question = (req.question or "").strip()

    metrics = {
        "trace_id": trace_id,
        "start_time": start_time,
        "end_time": "",
        "latency_ms": 0,
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

    logger.info(f"[trace_id={trace_id}] [chat] 收到请求 question_len={len(question)}，preview={question[:50]}")
    try:
        if not question:
            logger.warning(f"[trace_id={trace_id}] [chat] final_status=clarify reason=用户未输入问题")
            response.status_code = 400

            metrics["status"] = "no_relevant_info"
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

        decision = route_agent_action(question, final_top_k)
        action = decision.get("action")

        need_rag_ready_actions = {
            "search_knowledge_base",
            "list_documents",
            "get_chunk_text"
        }

        if action in need_rag_ready_actions:
            rag_status = get_rag_status()
            if not rag_status["ready"]:
                logger.warning(f"[trace_id={trace_id}] [chat] final_status=empty_index reason=知识库未准备完成")
                response.status_code = 503

                metrics["status"] = "internal_error"
                metrics["action"] = action or ""
                metrics["reason"] = "知识库未准备完成"

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
            result = run_agent_once(question, final_top_k, decision, trace_id)

            trace_metrics = collect_trace_metrics(result)

            metrics["status"] = normalize_metrics_status(result.get("status", ""))
            metrics["reference_count"] = len(result.get("references") or [])
            metrics["retrieved_count"] = trace_metrics["retrieved_count"]
            metrics["tool_called"] = trace_metrics["tool_called"]
            metrics["tool_success"] = trace_metrics["tool_success"]
            metrics["tool_name"] = trace_metrics["tool_name"]
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
        metrics["end_time"] = now_iso()
        metrics["latency_ms"] = int((time.perf_counter() - start_perf) * 1000)

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

        top1_vector_score = retrieved_chunks[0].get("vector_score", 0)
        if top1_vector_score < Settings.VECTOR_THRESHOLD:
            logger.warning(f"[search] top1_vector 分数过低 top1_vector_score={top1_vector_score}")
            response.status_code = 200
            return SearchResponse(
                code=0,
                message="success",
                data=SearchData(
                    question=question,
                    status="no_evidence",
                    reason=f"top1_vector 分数过低：{top1_vector_score:.4f}",
                    results=[]
                )
            )
        results = [
            SearchResult(
                reference_id=item["reference_id"],
                source_path=item["source_path"],
                file_name=item["file_name"],
                chunk_id=item["chunk_id"],
                position=item["position"],
                snippet=item["snippet"],
                score=item["hybrid_score"],
                bm25_score=item["bm25_score"],
                vector_score=item["vector_score"]
            )
            for item in retrieved_chunks
        ]

        score_brief = [
            {
                "chunk_id": item["chunk_id"],
                "score": round(item["hybrid_score"], 4),
                "bm25": round(item["bm25_score"], 4),
                "vector": round(item["vector_score"], 4),
            }
            for item in retrieved_chunks
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


@app.post("/extract_refund_policy", response_model=ExtractResponse)
def extract_api(req:ExtractRequest, response:Response):
    try:
        text = (req.text or "").strip()
        if not text:
            logger.warning("[extract] 空文本请求")
            response.status_code=400
            return ExtractResponse(
                code="EMPTY_TEXT",
                message="文本不能为空",
                data=ExtractData(
                    is_refundable=None,
                    refund_window_days=None,
                    exceptions=[],
                    evidence=[]
                )
            )

        data = extract_refund_policy(text)

        if data["status"] == "MODEL_BAD_OUTPUT":
            logger.warning("[extract] 模型输出格式异常")
            response.status_code = 200
            return ExtractResponse(
                code="MODEL_BAD_OUTPUT",
                message="模型输出格式异常，请稍后重试",
                data=ExtractData(
                    is_refundable=None,
                    refund_window_days=None,
                    exceptions=[],
                    evidence=[]
                )
            )

        logger.info("[extract] 请求处理完成")
        response.status_code = 200
        return ExtractResponse(
                code=0,
                message="success",
                data=ExtractData(
                    is_refundable=data.get("is_refundable", None),
                    refund_window_days=data.get("refund_window_days", None),
                    exceptions=data.get("exceptions", []),
                    evidence=data.get("evidence", [])
                )
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[extract] 未知异常 error={e}")
        response.status_code = 500
        return ExtractResponse(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            data=ExtractData(
                is_refundable=None,
                refund_window_days=None,
                exceptions=[],
                evidence=[]
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
            "internal_error_count": status_counter.get("internal_error", 0),
            "avg_latency_ms": avg_latency_ms,
        }
    }





















