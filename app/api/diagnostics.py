"""知识库诊断接口模块。

提供检索、文档预览、知识库摘要、路由决策和指标汇总等辅助接口。
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from fastapi import APIRouter, Query, Response

from app.agent.router import route_agent_action
from app.api.schemas import (
    ChatRequest,
    SearchData,
    SearchResponse,
    SearchResult,
)
from app.common.logging import get_logger
from app.config import AppSettings


logger = get_logger("chat_service")


# 集中保存诊断接口所需的配置、检索预览和指标读取依赖。
@dataclass(frozen=True)
class DiagnosticsDependencies:
    settings_provider: Callable[[], AppSettings]
    preview_docs: Callable
    preview_chunks: Callable
    get_rag_status: Callable
    run_diagnostic_search: Callable
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

    @router.post("/search", response_model=SearchResponse)
    def search_api(req: ChatRequest, response: Response):
        """清理检索输入，并将 RAG 诊断结果映射为 HTTP 响应。"""
        question = (req.question or "").strip()
        logger.info(
            "[search] request question_len=%s question_preview=%s",
            len(question),
            question[:50],
        )
        try:
            if not question:
                response.status_code = 400
                return SearchResponse(
                    code="EMPTY_QUESTION",
                    message="问题不能为空",
                    data=SearchData(
                        question=question,
                        status="error",
                        reason="用户未输入问题",
                        results=[],
                    ),
                )

            settings = deps.settings_provider()
            top_k = req.top_k if req.top_k is not None else settings.top_k
            diagnostic = deps.run_diagnostic_search(question, top_k)
            empty_index = diagnostic["status"] == "empty_index"
            results = [
                SearchResult(
                    reference_id=f"Reference {rank}",
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
                for rank, item in enumerate(diagnostic["chunks"], start=1)
            ]
            response.status_code = 503 if empty_index else 200
            return SearchResponse(
                code="EMPTY_INDEX" if empty_index else 0,
                message=diagnostic["message"],
                data=SearchData(
                    question=question,
                    status=diagnostic["status"],
                    reason=diagnostic["reason"],
                    results=results,
                ),
            )
        except Exception as exc:
            logger.exception("[search] processing failed error=%s", exc)
            response.status_code = 500
            return SearchResponse(
                code="INTERNAL_ERROR",
                message="服务器内部错误",
                data=SearchData(
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
        decision = route_agent_action(question)
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
                "model_error_count": statuses.get("model_error", 0),
                "clarify_count": statuses.get("clarify", 0),
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
