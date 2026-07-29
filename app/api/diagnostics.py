"""知识库诊断接口模块。

提供检索、文档预览、知识库摘要、路由决策和指标汇总等辅助接口。
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from fastapi import APIRouter, Query, Response

from app.api.schemas import ChatRequest
from app.common.logging import get_logger
from app.config import AppSettings


logger = get_logger("chat_service")


@dataclass(frozen=True)
class DiagnosticsDependencies:
    settings_provider: Callable[[], AppSettings]
    search_result: type
    search_data: type
    search_response: type
    preview_docs: Callable
    preview_chunks: Callable
    get_rag_status: Callable
    search_relevant_chunks: Callable
    rerank: Callable
    should_refuse: Callable
    safe_float: Callable
    route_node: Callable
    read_recent_metrics: Callable


def create_diagnostics_router(
    deps: DiagnosticsDependencies,
) -> APIRouter:
    """根据注入的依赖创建知识库与诊断接口路由。"""
    router = APIRouter()

    @router.get("/docs-preview")
    def docs_preview():
        """返回知识库文档注册信息的预览。"""
        docs = deps.preview_docs()
        return {"count": len(docs), "data": docs}

    @router.get("/chunks-preview")
    def chunks_preview(
        max_len: int = Query(default=300, ge=1, le=1000),
        limit: int = Query(70, ge=1, le=1000),
        file_name: str | None = Query(None),
    ):
        """按长度、数量和文件名筛选并返回 chunk 预览。"""
        data = deps.preview_chunks(max_len, limit, file_name)
        return {"count": len(data), "data": data}

    @router.post("/search", response_model=deps.search_response)
    def search_api(req: ChatRequest, response: Response):
        """执行知识库检索、重排和证据门控并返回候选片段。"""
        question = (req.question or "").strip()
        logger.info(
            "[search] request question_len=%s question_preview=%s",
            len(question),
            question[:50],
        )
        try:
            if not question:
                response.status_code = 400
                return deps.search_response(
                    code="EMPTY_QUESTION",
                    message="问题不能为空",
                    data=deps.search_data(
                        question=question,
                        status="error",
                        reason="用户未输入问题",
                        results=[],
                    ),
                )

            rag_status = deps.get_rag_status()
            if not rag_status["ready"]:
                response.status_code = 503
                return deps.search_response(
                    code="EMPTY_INDEX",
                    message=rag_status["message"],
                    data=deps.search_data(
                        question=question,
                        status="empty_index",
                        reason="知识库未准备完成",
                        results=[],
                    ),
                )

            settings = deps.settings_provider()
            top_k = req.top_k if req.top_k is not None else settings.top_k
            retrieved = deps.search_relevant_chunks(question, top_k)
            if not retrieved:
                return deps.search_response(
                    code=0,
                    message="success",
                    data=deps.search_data(
                        question=question,
                        status="no_evidence",
                        reason="未检索到结果",
                        results=[],
                    ),
                )

            final_chunks = deps.rerank(retrieved, question)[:top_k]
            if not final_chunks:
                return deps.search_response(
                    code=0,
                    message="success",
                    data=deps.search_data(
                        question=question,
                        status="no_evidence",
                        reason="重排后没有可用结果",
                        results=[],
                    ),
                )

            if deps.should_refuse(final_chunks):
                max_vector_score = max(
                    deps.safe_float(item.get("vector_score"))
                    for item in final_chunks
                )
                return deps.search_response(
                    code=0,
                    message="success",
                    data=deps.search_data(
                        question=question,
                        status="no_evidence",
                        reason=f"证据分数低于阈值：max_vector={max_vector_score:.4f}",
                        results=[],
                    ),
                )

            for rank, item in enumerate(final_chunks, start=1):
                item["reference_id"] = f"Reference {rank}"
            results = [
                deps.search_result(
                    reference_id=item["reference_id"],
                    source_path=item["source_path"],
                    file_name=item["file_name"],
                    chunk_id=item["chunk_id"],
                    position=item.get("position"),
                    snippet=item["snippet"],
                    score=item["hybrid_score"],
                    bm25_score=item["bm25_score"],
                    vector_score=item["vector_score"],
                    rerank_score=item.get("rerank_score", 0.0),
                )
                for item in final_chunks
            ]
            response.status_code = 200
            return deps.search_response(
                code=0,
                message="success",
                data=deps.search_data(
                    question=question,
                    status="ok",
                    reason="检索成功",
                    results=results,
                ),
            )
        except Exception as exc:
            logger.exception("[search] processing failed error=%s", exc)
            response.status_code = 500
            return deps.search_response(
                code="INTERNAL_ERROR",
                message="服务器内部错误",
                data=deps.search_data(
                    question=question,
                    status="error",
                    reason="search 接口处理异常",
                    results=[],
                ),
            )

    @router.post("/kb_summary")
    def kb_summary():
        """汇总知识库状态、文档数量和文件名称。"""
        status = deps.get_rag_status()
        docs = deps.preview_docs()
        return {
            "code": 0,
            "message": "success",
            "data": {
                "ready": status.get("ready"),
                "message": status.get("message"),
                "doc_count": status.get("doc_count"),
                "chunk_count": status.get("chunk_count"),
                "index_source": status.get("index_source", "unknown"),
                "file_names": [doc["file_name"] for doc in docs],
                "documents": docs,
            },
        }

    @router.post("/decide")
    def decide_api(req: ChatRequest, response: Response):
        """返回 Agent 对给定问题作出的路由决策。"""
        question = (req.question or "").strip()
        if not question:
            response.status_code = 400
            return {
                "code": "EMPTY_QUESTION",
                "message": "问题不能为空",
                "data": {
                    "question": question,
                    "decision": "clarify",
                    "reason": "用户未输入问题",
                },
            }
        route_output = deps.route_node({"effective_question": question})
        decision = route_output["decision"]
        return {
            "code": 0,
            "message": "success",
            "data": {"question": question, "decision": decision},
        }

    @router.get("/metrics-summary")
    def metrics_summary(limit: int = Query(20, ge=1, le=200)):
        """统计最近请求的状态、耗时和 LLM 调用情况。"""
        records = deps.read_recent_metrics(limit)
        statuses = Counter(record.get("status", "unknown") for record in records)
        count = len(records)
        average = lambda key: round(
            sum(record.get(key, 0) for record in records) / count, 2
        ) if count else 0
        return {
            "code": 0,
            "message": "success",
            "data": {
                "limit": limit,
                "total": count,
                "status_count": dict(statuses),
                "success_count": statuses.get("success", 0),
                "no_relevant_info_count": statuses.get("no_relevant_info", 0),
                "tool_error_count": statuses.get("tool_error", 0),
                "model_bad_output_count": statuses.get("model_bad_output", 0),
                "model_error_count": statuses.get("model_error", 0),
                "clarify_count": statuses.get("clarify", 0),
                "out_of_scope_count": statuses.get("out_of_scope", 0),
                "empty_index_count": statuses.get("empty_index", 0),
                "internal_error_count": statuses.get("internal_error", 0),
                "avg_latency_ms": average("latency_ms"),
                "avg_llm_call_count": average("llm_call_count"),
                "router_llm_used_count": sum(
                    bool(record.get("router_llm_used")) for record in records
                ),
                "decompose_llm_used_count": sum(
                    bool(record.get("decompose_llm_used")) for record in records
                ),
                "answer_llm_used_count": sum(
                    bool(record.get("answer_llm_used")) for record in records
                ),
            },
        }

    return router
