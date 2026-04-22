from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field, ConfigDict
from app.rag import (
    rebuild_index,
    ask,
    preview_docs,
    preview_chunks,
    search_relevant_chunks,
    get_rag_status,
    needs_rebuild,
    save_manifest,
    build_current_manifest,
)
from app.config import Settings
from app.untils import get_logger
from app.extract_refund_policy import extract_refund_policy
from typing import Literal, TypeAlias
from app.tools import dispatch_tool
import re

AnswerStatus: TypeAlias = Literal[
    "answered",
    "insufficient",
    "clarify",
    "out_of_scope",
    "error",
]
app = FastAPI(title='Mini Rag API', version='1.0.0')
logger = get_logger('chat_service')

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
    answer_status:Literal["answered", "insufficient", "clarify", "out_of_scope", "error"]
    clarification_question:str = Field(..., description="补充提示")
    reason:str = Field(..., description="理由")

class ChatResponse(BaseModel):
    code:int|str
    message:str
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
    results:list[SearchResult]=Field(default_factory=list)

class SearchResponse(BaseModel):
    code:int
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

def route_tool_request(question: str):
    q = (question or "").strip()

    if "文档" in q and any(x in q for x in ["几个", "多少", "数量", "列表", "有哪些"]):
        return {
            "tool_name": "list_documents",
            "tool_args": {}
        }

    m = re.search(r'chunk[_\s\-:：]*([0-9]+)', q, re.IGNORECASE)
    if m and any(x in q for x in ["原文", "内容", "查看", "给我看", "显示"]):
        return {
            "tool_name": "get_chunk_text",
            "tool_args": {
                "chunk_id": int(m.group(1))
            }
        }

    return None

def answer_with_tools_or_rag(question: str, top_k: int) -> dict:
    intent = route_tool_request(question)

    # 1) 命中工具：直接走工具
    if intent is not None:
        tool_name = intent["tool_name"]
        tool_args = intent["tool_args"]

        logger.info(
            f"[route] mode=tool tool_name={tool_name} "
            f"question={question[:50]} tool_args={tool_args}"
        )

        tool_result = dispatch_tool(tool_name, tool_args)

        if tool_name == "list_documents":
            docs = tool_result.get("documents", [])
            count = tool_result.get("count", 0)

            doc_names = []
            for d in docs[:10]:
                doc_names.append(
                    d.get("source_file")
                    or d.get("file_name")
                    or d.get("title")
                    or d.get("doc_name")
                    or d.get("doc_id")
                    or "未命名文档"
                )

            answer = f"当前系统中共有 {count} 个文档。"
            if doc_names:
                answer += " 文档包括：" + "、".join(doc_names)

            return {
                "status": "OK",
                "answer": answer,
                "references": [],
                "confidence": 1.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": "命中文档列表工具"
            }

        if tool_name == "get_chunk_text":
            if not tool_result.get("found"):
                return {
                    "status": "NO_RELEVANT_INFO",
                    "answer": f"没有找到 chunk_id 为 {tool_args.get('chunk_id')} 的内容。",
                    "references": [],
                    "confidence": 0.0,
                    "need_clarification": False,
                    "clarification_question": "",
                    "reason": "chunk_id 不存在"
                }

            return {
                "status": "OK",
                "answer": f"这是 {tool_args.get('chunk_id')} 的原文：\n\n{tool_result.get('text', '')}",
                "references": [],
                "confidence": 1.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": "命中 chunk 原文工具"
            }

        return {
            "status": "MODEL_BAD_OUTPUT",
            "answer": "",
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "工具执行后未命中预期分支"
        }

    # 2) 没命中工具：继续走 RAG
    logger.info(
        f"[route] mode=rag top_k={top_k} question={question[:50]}"
    )

    logger.info(f"[retrieve] 开始检索 top_k={top_k}")
    retrieved_chunks = search_relevant_chunks(question, top_k)
    hit_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
    logger.info(f"[retrieve] 检索完成 chunks_len={len(hit_ids)}，hit_ids={hit_ids}")

    logger.info("[answer] 开始调用模型")
    result = ask(question, top_k=top_k, retrieved_chunks=retrieved_chunks)

    return result

@app.on_event("startup")
def start_on_event():
    Settings.validate()

    data_dir = str(Settings.RAG_FILE_PATH)

    # 这里只做记录，不阻止重建
    need, reason = needs_rebuild(data_dir)
    logger.info(f"[startup] needs_rebuild={need}, reason={reason}")

    rebuild_index(
        data_dir,
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP
    )

    save_manifest(build_current_manifest(data_dir))
    logger.info("[startup] 索引已加载到内存，并已保存 manifest")

OUT_OF_SCOPE_KEYWORDS = [
    "天气", "气温", "下雨",
    "写诗", "作诗", "故事", "笑话",
    "股票", "基金", "彩票",
    "电影", "明星", "八卦",
    "数学题", "微积分", "量子力学",
    "python教程", "java教程", "怎么写代码"
]

def is_out_of_scope(question:str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(keyword in q for keyword in OUT_OF_SCOPE_KEYWORDS)

def build_chat_response(code:int|str,
                        message:str,
                        question:str,
                        answer:str="",
                        references:list | None = None,
                        confidence: float = 0.0,
                        need_clarification: bool = False,
                        answer_status:AnswerStatus = "answered",
                        clarification_question:str = "",
                        reason:str = ""
                        ) -> ChatResponse:
    return ChatResponse(code=code,
                        message=message,
                        data=ChatData(
                            question=question,
                            answer=answer,
                            references=references or [],
                            confidence=confidence,
                            need_clarification=need_clarification,
                            answer_status=answer_status,
                            clarification_question=clarification_question,
                            reason=reason
                            )
                        )

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
    question = (req.question or "").strip()
    logger.info(f"[chat] 收到请求 question_len={len(question)}，preview={question[:50]}")
    try:
        if not question:
            logger.warning("[chat] 空问题请求")
            response.status_code = 400
            return build_chat_response(
                code="EMPTY_QUESTION",
                message="问题不能为空，请输入你想查询的内容",
                question=question,
                confidence=0.0,
                need_clarification=True,
                answer_status="clarify",
                clarification_question="请直接输入你想查询的具体问题，例如：年假申请流程是什么？",
                reason="用户未输入问题"
            )

        if is_out_of_scope(question):
            response.status_code = 200
            return build_chat_response(
                code="OUT_OF_SCOPE",
                message="这个问题不在当前知识库回答范围内",
                question=question,
                answer="这个问题不属于当前知识库覆盖范围，我目前只能根据已收录资料回答相关问题。",
                references=[],
                confidence=0.0,
                need_clarification=False,
                answer_status="out_of_scope",
                clarification_question="",
                reason="问题主题与知识库无关"
            )

        rag_status = get_rag_status()
        if not rag_status["ready"]:
            logger.warning("[chat] 数据目录为空")
            response.status_code = 503
            return build_chat_response(
                code="NO_DOCUMENTS",
                message=rag_status["message"],
                question=question,
                confidence=0.0,
                need_clarification=False,
                answer_status="insufficient",
                clarification_question="",
                reason="知识库未准备完成"
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K

        try:
            logger.info("[answer] 开始处理：工具优先，否则走 RAG")
            result = answer_with_tools_or_rag(question, final_top_k)

        except Exception as e:
            logger.exception(f"[answer] 处理失败 error={e}")
            response.status_code = 500
            return build_chat_response(
                code="MODEL_ERROR",
                message="处理失败，请稍后重试",
                question=question,
                confidence=0.0,
                need_clarification=False,
                answer_status="insufficient",
                clarification_question="",
                reason="工具调用或模型调用异常"
            )

        if result["status"] == "NEED_CLARIFICATION":
            response.status_code = 200
            return build_chat_response(
                code="NEED_CLARIFICATION",
                message="问题信息不足，需要补充后再回答",
                question=question,
                answer="",
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=True,
                answer_status="clarify",
                clarification_question=result.get("clarification_question", ""),
                reason=result.get("reason", "问题缺少关键限定信息")
            )

        if result["status"] == "NO_RELEVANT_INFO":
            logger.warning("[chat] 检索证据不足，返回 honest fallback")
            response.status_code = 200
            return build_chat_response(
                code="NO_RELEVANT_INFO",
                message="我没有在文档中找到相关信息",
                question=question,
                answer=result.get("answer", ""),
                references=result.get("references", []),
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                answer_status="insufficient",
                clarification_question="",
                reason=result.get("reason", "模型判断当前上下文不足以支撑回答")
            )

        if result["status"] == "MODEL_BAD_OUTPUT":
            logger.warning("[chat] 模型输出格式异常")
            response.status_code = 200
            return build_chat_response(
                code="MODEL_BAD_OUTPUT",
                message="模型输出格式异常，请稍候重试",
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                answer_status="error",
                clarification_question="",
                reason=result.get("reason", "模型返回结构不符合预期")
            )

        logger.info("[chat] 请求处理完成")
        response.status_code = 200
        return build_chat_response(
            code=0,
            message="success",
            question=question,
            answer=result["answer"],
            references=result.get("references", []),
            confidence=result.get("confidence", 0.0),
            need_clarification=result.get("need_clarification", False),
            answer_status="answered" if result["status"] == "OK" else "error",
            clarification_question=result.get("clarification_question", ""),
            reason=result.get("reason", "正常回答")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[chat] 未知异常 error={e}")
        response.status_code = 500
        return build_chat_response(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            question=question,
            confidence=0.0,
            need_clarification=False
        )

@app.get("/docs-preview")
def docs_preview():
    docs = preview_docs()
    return {"count":len(docs), "data":docs}

@app.get("/chunks-preview")
def chunks_preview(
        limit:int = Query(70, ge=1, le=1000),
        file_name:str | None=Query(None)):
    data = preview_chunks(limit, file_name)
    return {"count":len(data), "data":data}

@app.post("/search", response_model=SearchResponse)
def search_api(req:ChatRequest):
    try:
        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K
        retrieved_chunks = search_relevant_chunks(req.question, final_top_k)

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

        return SearchResponse(
            code=0,
            message="success",
            data=SearchData(
                question=req.question,
                results=results
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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





















