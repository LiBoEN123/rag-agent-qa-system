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
    needs_rebuild,
    load_index_cache,
    index_cache_exists,
    rebuild_and_persist_index,
)
from app.config import Settings
from app.untils import get_logger
from app.agent import route_agent_action, run_agent_once
from app.observability import create_trace_id
from app.rag_retrieval import safe_float, should_refuse

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
    decompose_top_k:int | None=Field(None, ge=1, le=10, description='召回条数')


    model_config = ConfigDict(
        json_schema_extra={
            "example":{
              "question": "普通本科四年制专业学生现在最长学习年限是不是 8 年？",
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
    retrieval_confidence:float = Field(..., ge=0.0, le=1.0, description="检索证据置信度，0~1")
    answer_confidence:float = Field(..., ge=0.0, le=1.0, description="最终回答置信度，0~1")
    need_clarification:bool = Field(..., description="是否建议用户补充问题")
    status: PublicStatus
    clarification_question:str = Field(..., description="补充提示")
    reason:str = Field(..., description="理由")
    action:str = Field(default="", description="本次Agent选择的动作")
    tool_result_summary:str = Field(default="", description="工具结果摘要")
    tool_trace:list[dict] = Field(default_factory=list, description="工具调用或路径追踪")
    router_llm_used: bool = Field(default=False, description="本次请求是否调用了 Router LLM")
    decompose_llm_used: bool = Field(default=False, description="本次请求是否调用了问题拆分 LLM")
    answer_llm_used: bool = Field(default=False, description="本次请求是否调用了最终回答 LLM")
    llm_call_count: int = Field(default=0, ge=0, description="本次请求累计 LLM 调用次数")

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
                "trace_id": "req_3b745087",
                "data": {
                    "question": "普通本科四年制专业学生现在最长学习年限是不是 8 年？",
                    "answer": "根据参考资料，普通本科四年制专业学生的最长学习年限因学生类型而异：一般学生（非休学创业、非应征入伍）的学习年限为3～6年（见Reference 1和Reference 2），而休学创业学生的最长学习年限为8年（见Reference 1和Reference 2）。因此，对于普通本科四年制专业的一般学生，最长学习年限不是8年，而是6年；对于休学创业学生，最长学习年限是8年。",
                    "references": [
                    {
                        "reference_id": "Reference 1",
                        "file_name": "内蒙古工业大学教务处你问我答栏目整理.md",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学教务处你问我答栏目整理.md",
                        "chunk_id": 371,
                        "position": 45,
                        "snippet": "[标题路径]内蒙古工业大学教务处“你问我答”栏目整理>38.我们专业的修业年限是几年？**发布日期：**2024-08-31**原网页：**https://jwch.imut.edu.cn/info/1060/5796.htm答：从2023年入学的学生开始，普通本科四年制专业学生学习年限为3～6年；普通本科五年制专业学生学习年限为4～7年；休学创业学生最长学习年限为8年。应征入伍学生服役阶段不计入学习年限。---",
                        "score": 0.9674,
                        "rerank_score": 0.9892
                    },
                    {
                        "reference_id": "Reference 2",
                        "file_name": "内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "chunk_id": 7,
                        "position": 2,
                        "snippet": "[标题路径]内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知>一、调整修业年限对《内蒙古工业大学学籍管理办法》第四章第十三条调整如下：学生根据自身实际情况，可适当缩短或延长学习年限。普通本科四年制专业学生学习年限调整为3～6年；普通本科五年制专业学生学习年限调整为4～7年；休学创业学生最长学习年限调整为8年。应征入伍学生服役阶段不计入学习年限。",
                        "score": 0.9426,
                        "rerank_score": 0.9817
                    }
                    ],
                    "retrieval_confidence": 0.812,
                    "answer_confidence": 0.546,
                    "need_clarification": False,
                    "status": "ok",
                    "clarification_question": "",
                    "reason": "检索命中有效证据，且答案成功绑定引用",
                    "action": "search_knowledge_base",
                    "tool_result_summary": "知识库检索完成，召回 5 个 chunk",
                    "tool_trace": [
                    {
                        "step": "decide",
                        "route_decision": "rag",
                        "decision_type": "use_knowledge_base",
                        "action": "search_knowledge_base",
                        "tool_name": "",
                        "tool_args": {
                        "query": "普通本科四年制专业学生现在最长学习年限是不是 8 年?",
                        "top_k": 5
                        },
                        "reason": "用户询问学制年限，属于知识库内容，需要检索相关文档。",
                        "router_source": "llm",
                        "router_confidence": 0.95,
                        "router_fallback_reason": ""
                    },
                    {
                        "step": "call_tool",
                        "tool_name": "search_knowledge_base",
                        "tool_args": {
                        "query": "普通本科四年制专业学生现在最长学习年限是不是 8 年？",
                        "top_k": 5
                        },
                        "tool_status": "success",
                        "success": True,
                        "final_chunk_count": 5,
                        "top_chunk_ids": [
                        371,
                        7,
                        228,
                        310,
                        6
                        ],
                        "top_scores": [
                        0.9674,
                        0.9426,
                        0.8981,
                        0.8352,
                        0.6181
                        ],
                        "result_summary": "检索到 5 个候选 chunk",
                        "hits": [
                        {
                            "chunk_id": 371,
                            "rerank": 0.9892,
                            "score": 0.9674,
                            "bm25": 40.8263,
                            "vector": 0.737,
                            "file_name": "内蒙古工业大学教务处你问我答栏目整理.md"
                        },
                        {
                            "chunk_id": 7,
                            "rerank": 0.9817,
                            "score": 0.9426,
                            "bm25": 40.9276,
                            "vector": 0.7034,
                            "file_name": "内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md"
                        },
                        {
                            "chunk_id": 228,
                            "rerank": 0.9816,
                            "score": 0.8981,
                            "bm25": 31.8092,
                            "vector": 0.7778,
                            "file_name": "内蒙古工业大学学生手册.md"
                        },
                        {
                            "chunk_id": 310,
                            "rerank": 0.9149,
                            "score": 0.8352,
                            "bm25": 31.0278,
                            "vector": 0.7076,
                            "file_name": "内蒙古工业大学学籍管理办法（2018年修订）.md"
                        },
                        {
                            "chunk_id": 6,
                            "rerank": 0.5652,
                            "score": 0.6181,
                            "bm25": 17.6933,
                            "vector": 0.6194,
                            "file_name": "内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md"
                        }
                        ]
                    },
                    {
                        "step": "format_context",
                        "summary": "将 5 个 chunk 交给 RAG 回答逻辑"
                    },
                    {
                        "step": "final_answer",
                        "tool_status": "skipped",
                        "final_status": "ok",
                        "answer_preview": "根据参考资料，普通本科四年制专业学生的最长学习年限因学生类型而异：一般学生（非休学创业、非应征入伍）的学习年限为3～6年（见Reference 1和Refere"
                    }
                    ]
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
                    "question": "普通本科四年制专业学生现在最长学习年限是不是 8 年？",
                    "status": "ok",
                    "reason": "检索成功",
                    "results": [
                    {
                        "reference_id": "Reference 1",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学教务处你问我答栏目整理.md",
                        "file_name": "内蒙古工业大学教务处你问我答栏目整理.md",
                        "chunk_id": 371,
                        "position": 45,
                        "snippet": "[标题路径]内蒙古工业大学教务处“你问我答”栏目整理>38.我们专业的修业年限是几年？**发布日期：**2024-08-31**原网页：**https://jwch.imut.edu.cn/info/1060/5796.htm答：从2023年入学的学生开始，普通本科四年制专业学生学习年限为3～6年；普通本科五年制专业学生学习年限为4～7年；休学创业学生最长学习年限为8年。应征入伍学生服役阶段不计入学习年限。---",
                        "score": 0.9674021005630493,
                        "bm25_score": 40.82631854778893,
                        "vector_score": 0.7369934320449829,
                        "rerank_score": 0.9891571998596191
                    },
                    {
                        "reference_id": "Reference 2",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "file_name": "内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "chunk_id": 7,
                        "position": 2,
                        "snippet": "[标题路径]内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知>一、调整修业年限对《内蒙古工业大学学籍管理办法》第四章第十三条调整如下：学生根据自身实际情况，可适当缩短或延长学习年限。普通本科四年制专业学生学习年限调整为3～6年；普通本科五年制专业学生学习年限调整为4～7年；休学创业学生最长学习年限调整为8年。应征入伍学生服役阶段不计入学习年限。",
                        "score": 0.9426043033599854,
                        "bm25_score": 40.927641430801266,
                        "vector_score": 0.7033796310424805,
                        "rerank_score": 0.9816882014274597
                    },
                    {
                        "reference_id": "Reference 3",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学学生手册.md",
                        "file_name": "内蒙古工业大学学生手册.md",
                        "chunk_id": 228,
                        "position": 219,
                        "snippet": "[标题路径]学籍管理>第四章修业年限第十二条全日制普通本科专业基本学制为4年或5年，全日制高职高专学生基本学制为3年。第十三条学生根据自身实际情况，可适当缩短或延长修业年限。一般普通本科四年制专业学生修业年限为3～8年；普通本科五年制专业学生修业年限为4～10年；高职高专学生修业年限为2～6年。应征入伍学生修业年限为学制×2+服役年数。休学创业学生最长修业年限为10年。第十四条修业年限计算时间自学生正式获得我校学籍起，至学历证书颁发时间止。休学及保留学籍时间包含在修业年限内。第十五条学生可以分阶段完成学业，超过规定年限者，学校不予注册，不再保留其学籍。",
                        "score": 0.8980649709701538,
                        "bm25_score": 31.809159039335132,
                        "vector_score": 0.7777819037437439,
                        "rerank_score": 0.9815797209739685
                    },
                    {
                        "reference_id": "Reference 4",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学学籍管理办法（2018年修订）.md",
                        "file_name": "内蒙古工业大学学籍管理办法（2018年修订）.md",
                        "chunk_id": 310,
                        "position": 7,
                        "snippet": "[标题路径]内蒙古工业大学学籍管理办法>第四章修业年限**第十二条**全日制普通本科专业基本学制为4年或5年，全日制高职高专学生基本学制为3年。**第十三条**学生根据自身实际情况，可适当缩短或延长修业年限。一般普通本科四年制专业学生修业年限为3～8年；普通本科五年制专业学生修业年限为4～10年；高职高专学生修业年限为2～6年。应征入伍学生修业年限为学制×2+服役年数。休学创业学生最长修业年限为10年。**第十四条**修业年限计算时间自学生正式获得我校学籍起，至学历证书颁发时间止。休学及保留学籍时间包含在修业年限内。**第十五条**学生可以分阶段完成学业，超过规定年限者，学校不予注册，不再保留",
                        "score": 0.8352251648902893,
                        "bm25_score": 31.027838157205085,
                        "vector_score": 0.7076448202133179,
                        "rerank_score": 0.9148778915405273
                    },
                    {
                        "reference_id": "Reference 5",
                        "source_path": "/home/wabce94123/projects/rag-agent-qa-system/data/kb/内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "file_name": "内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知.md",
                        "chunk_id": 6,
                        "position": 1,
                        "snippet": "[标题路径]内蒙古工业大学关于调整普通全日制本科学生最长学习年限和成绩记载方式的通知为近一步提高人才培养质量，促进学生学习主动性，根据教育部《普通高等学校学生管理规定》（2017年教育部令第41号），经学校研究，决定对《内蒙古工业大学学籍管理办法》（内工大校发〔2018〕62号）中修业年限和成绩记载方式相关规定进行调整。",
                        "score": 0.6181007623672485,
                        "bm25_score": 17.69333164248942,
                        "vector_score": 0.6194202899932861,
                        "rerank_score": 0.5651710629463196
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
    """根据 error_type 判断记录是否应该写入 bad_cases.jsonl。"""
    error_type = str(record.get("error_type", "") or "").strip()

    if error_type in REAL_BAD_CASE:
        return True

    if error_type in EXPECTED_NON_BAD_CASE_TYPES:
        return False

    # 防御：如果以后出现了未知 error_type，先不要静默写入。
    logger.info(f"[trace_id={record.get('trace_id', '')}] [bad_case] skip unknown error_type={error_type}")

    return False

def now_iso() -> str:
    """生成带毫秒精度的当前时间字符串。"""
    return datetime.now().isoformat(timespec="milliseconds")

@app.middleware("http")
async def trace_and_timing_middleware(request: Request, call_next):
    """为 /chat 请求注入 trace_id 并记录请求总耗时。"""
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

    """将内部大写状态转换成 metrics.jsonl 使用的统计状态。"""
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

    """从 Agent 的 tool_trace 中提取检索、工具调用和耗时信息。"""
    tool_trace = result.get("tool_trace") or []

    retrieved_count = 0
    tool_called = False
    tool_success = False
    tool_name = ""
    router_llm_used = bool(result.get("router_llm_used", False))
    decompose_llm_used = bool(result.get("decompose_llm_used", False))
    answer_llm_used = bool(result.get("answer_llm_used", False))
    llm_call_count = result.get("llm_call_count")

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

        if step.get("step") == "llm_usage":
            router_llm_used = bool(step.get("router_llm_used", router_llm_used))
            decompose_llm_used = bool(step.get("decompose_llm_used", decompose_llm_used))
            answer_llm_used = bool(step.get("answer_llm_used", answer_llm_used))
            llm_call_count = step.get("llm_call_count", llm_call_count)

    if llm_call_count is None:
        llm_call_count = (
            int(router_llm_used)
            + int(decompose_llm_used)
            + int(answer_llm_used)
        )

    return {
        "retrieved_count": retrieved_count,
        "tool_called": tool_called,
        "tool_success": tool_success,
        "tool_name": tool_name,
        "router_llm_used": router_llm_used,
        "decompose_llm_used": decompose_llm_used,
        "answer_llm_used": answer_llm_used,
        "llm_call_count": int(llm_call_count or 0),
        "timings": result.get("timings") or {},
    }

def write_metrics(record: dict) -> None:

    """将一条指标记录追加写入 metrics.jsonl。"""
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

    """打印指标日志并持久化到 metrics.jsonl。"""
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
        f"router_llm_used={record.get('router_llm_used')} "
        f"decompose_llm_used={record.get('decompose_llm_used')} "
        f"answer_llm_used={record.get('answer_llm_used')} "
        f"llm_call_count={record.get('llm_call_count')} "
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
                        retrieval_confidence: float = 0.0,
                        answer_confidence: float = 0.0,
                        need_clarification: bool = False,
                        status: PublicStatus = "ok",
                        clarification_question:str = "",
                        reason:str = "",
                        action:str = "",
                        tool_result_summary:str = "",
                        tool_trace:list | None = None,
                        router_llm_used: bool | None = None,
                        decompose_llm_used: bool | None = None,
                        answer_llm_used: bool | None = None,
                        llm_call_count: int | None = None,
                        ) -> ChatResponse:
    """按统一响应模型组装 /chat 返回数据。"""
    retrieval_confidence = max(0.0, min(1.0, safe_float(retrieval_confidence)))
    answer_confidence = max(0.0, min(1.0, safe_float(answer_confidence)))
    tool_trace = tool_trace or []

    if (
        router_llm_used is None
        or decompose_llm_used is None
        or answer_llm_used is None
        or llm_call_count is None
    ):
        for step in tool_trace:
            if step.get("step") == "llm_usage":
                router_llm_used = bool(step.get("router_llm_used", router_llm_used or False))
                decompose_llm_used = bool(step.get("decompose_llm_used", decompose_llm_used or False))
                answer_llm_used = bool(step.get("answer_llm_used", answer_llm_used or False))
                llm_call_count = step.get("llm_call_count", llm_call_count)
                break

    router_llm_used = bool(router_llm_used)
    decompose_llm_used = bool(decompose_llm_used)
    answer_llm_used = bool(answer_llm_used)
    if llm_call_count is None:
        llm_call_count = (
            int(router_llm_used)
            + int(decompose_llm_used)
            + int(answer_llm_used)
        )

    return ChatResponse(code=code,
                        message=message,
                        trace_id=trace_id,
                        data=ChatData(
                            question=question,
                            answer=answer,
                            references=references or [],
                            retrieval_confidence=retrieval_confidence,
                            answer_confidence=answer_confidence,
                            need_clarification=need_clarification,
                            status=status,
                            clarification_question=clarification_question,
                            reason=reason,
                            action=action,
                            tool_result_summary=tool_result_summary,
                            tool_trace=tool_trace,
                            router_llm_used=router_llm_used,
                            decompose_llm_used=decompose_llm_used,
                            answer_llm_used=answer_llm_used,
                            llm_call_count=max(0, int(llm_call_count or 0))
                            )
                        )

def get_normalize_status(result: dict) -> str:
    """将 Agent 结果状态映射为对外公开的响应状态。"""
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
    """记录最终回答状态、引用数量和是否需要澄清。"""
    answer_status = get_normalize_status(result)
    reference_count = len(result.get("references") or [])
    need_clarification = result.get("need_clarification", False)

    logger.info(
        f"[trace_id={trace_id}] [final] answer_status={answer_status} "
        f"reference_count={reference_count} need_clarification={need_clarification}"
    )

def read_recent_metrics(limit: int = 20) -> list[dict]:

    """读取最近若干条 metrics.jsonl 记录供汇总接口使用。"""
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
    """FastAPI 启动时校验配置并加载或重建 RAG 索引。"""
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
        "router_llm_used": False,
        "decompose_llm_used": False,
        "answer_llm_used": False,
        "llm_call_count": 0,
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
                need_clarification=True,
                status="clarify",
                clarification_question="请直接输入你想查询的具体问题，例如：年假申请流程是什么？",
                reason="用户未输入问题"
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K
        decompose_top_k = req.decompose_top_k if req.decompose_top_k is not None else Settings.DECOMPOSE_TOP_K

        route_start = time.perf_counter()

        decision = route_agent_action(question, final_top_k, trace_id=trace_id)

        route_cost_ms = int((time.perf_counter() - route_start) * 1000)
        metrics["route_cost_ms"] = route_cost_ms

        action = decision.get("action")
        metrics["router_llm_used"] = bool(
            decision.get(
                "router_llm_used",
                decision.get("router_source") in {"llm", "rule_based_fallback"}
            )
        )
        metrics["llm_call_count"] = int(metrics["router_llm_used"])

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
                decompose_top_k,
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
            metrics["llm_call_count"] = int(metrics["router_llm_used"])
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
                retrieval_confidence=result.get("retrieval_confidence", 0.0),
                answer_confidence=result.get("answer_confidence", 0.0),
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
            retrieval_confidence=result.get("retrieval_confidence", 0.0),
            answer_confidence=result.get("answer_confidence", 0.0),
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
            f"llm_cost_ms={metrics.get('llm_cost_ms')} "
            f"llm_call_count={metrics.get('llm_call_count')}"
        )

        log_and_write_metrics(metrics)

@app.get("/docs-preview")
def docs_preview():
    """返回知识库文档注册表预览。"""
    docs = preview_docs()
    return {"count":len(docs), "data":docs}

@app.get("/chunks-preview")
def chunks_preview(
        max_len:int = Query(default=300, ge=1, le=1000),
        limit:int = Query(70, ge=1, le=1000),
        file_name:str | None=Query(None)):
    """返回知识库 chunk 的简短预览列表。"""
    data = preview_chunks(max_len,limit, file_name)
    return {"count":len(data), "data":data}

@app.post("/search", response_model=SearchResponse)
def search_api(req:ChatRequest, response:Response):
    """处理 /search 请求，只执行检索和重排并返回候选片段。"""
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

        if should_refuse(final_chunks):
            max_vector_score = max(
                safe_float(item.get("vector_score"))
                for item in final_chunks
            )
            logger.warning(f"[search] 检索证据不足 max_vector_score={max_vector_score}")
            response.status_code = 200
            return SearchResponse(
                code=0,
                message="success",
                data=SearchData(
                    question=question,
                    status="no_evidence",
                    reason=f"证据分数低于阈值：max_vector={max_vector_score:.4f}",
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
    """返回知识库就绪状态、文档数量和文档列表。"""
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
            "index_source": status.get("index_source", "unknown"),
            "file_names":[doc["file_name"] for doc in docs],
            "documents":docs
        }
    }

@app.post("/decide")
def decide_api(req:ChatRequest, response:Response):
    """返回给定问题的 Agent 路由决策，便于调试。"""
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
    """汇总最近的指标记录，返回状态分布和平均延迟。"""
    records = read_recent_metrics(limit)

    status_counter = Counter(record.get("status", "unknown") for record in records)

    if records:
        avg_latency_ms = round(
            sum(record.get("latency_ms", 0) for record in records) /
            len(records), 2)
        avg_llm_call_count = round(
            sum(record.get("llm_call_count", 0) for record in records) /
            len(records), 2)

    else:
        avg_latency_ms = 0
        avg_llm_call_count = 0

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
            "avg_llm_call_count": avg_llm_call_count,
            "router_llm_used_count": sum(
                1 for record in records if record.get("router_llm_used")
            ),
            "decompose_llm_used_count": sum(
                1 for record in records if record.get("decompose_llm_used")
            ),
            "answer_llm_used_count": sum(
                1 for record in records if record.get("answer_llm_used")
            ),
        }
    }
