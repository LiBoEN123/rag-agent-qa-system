from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field, ConfigDict
from app.rag import (
    ask,
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
from typing import Literal, TypeAlias
from app.tools import dispatch_tool
import re

PublicStatus: TypeAlias = Literal[
    "ok",
    "no_evidence",
    "empty_index",
    "error",
    "clarify",
    "out_of_scope",
]

OUT_OF_SCOPE_KEYWORDS = [
    "天气", "气温", "下雨",
    "写诗", "作诗", "故事", "笑话",
    "股票", "基金", "彩票",
    "电影", "明星", "八卦",
    "数学题", "微积分", "量子力学",
    "python教程", "java教程", "怎么写代码"
]
app = FastAPI(title='Mini Rag API', version='1.0.0')
logger = get_logger('chat_service')

# =========================
# 功能性函数定义
# =========================

# Agent决策层：判断走哪个动作
def route_agent_action(question: str, top_k:int) -> dict:
    q = (question or "").strip()

    # 1. 空问题：需要澄清
    if not q:
        return {
            "action": "clarify",
            "tool_args": {},
            "reason": "用户未输入问题",
            "clarification_question": "请直接输入你想查询的具体问题，例如：每个 Vue 应用是通过什么函数创建实例的？"
        }

    # 2. 用户问系统能力：可以直接回答，不需要查知识库
    if q in ["你是谁?", "你能做什么?", "你可以做什么?", "你有什么功能?"]:
        return {
            "action": "direct_answer",
            "tool_args": {},
            "reason": "用户询问系统能力，可以直接回答"
        }

    # 3. 用户问知识库文档数量或列表：调用 list_documents
    if "文档" in q and any(x in q for x in ["几个", "多少", "数量", "列表", "有哪些"]):
        return {
            "action": "list_documents",
            "tool_args": {},
            "reason": "用户询问知识库文档列表或文档数量"
        }

    # 4. 用户输入简单数学表达式：调用 simple_calculator
    expression = extract_simple_math_expression(question)
    if expression is not None:
        return {
            "action": "simple_calculator",
            "tool_args": {
                "expression": expression
            },
            "reason": "用户问题是简单四则运算"
        }

    # 5. 明显超出知识库范围：拒答
    if is_out_of_scope(q):
        return {
            "action": "refuse",
            "tool_args": {},
            "reason": "问题主题超出当前知识库范围"
        }

    # 6. 指代太模糊：澄清
    if q in ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"]:
        return {
            "action": "clarify",
            "tool_args": {},
            "reason": "问题存在明显指代不明",
            "clarification_question": "你说的“这个”具体指哪个功能、哪份文档或哪段报错？请补充一下对象。"
        }

    # 7. 默认：当成知识库问题，走 RAG
    return {
        "action": "search_knowledge_base",
        "tool_args": {
            "query": q,
            "top_k": top_k
        },
        "reason": "默认作为知识库问答问题处理"
    }

def run_agent_once(question: str, top_k: int) -> dict:
    decision = route_agent_action(question, top_k)
    action = decision['action']

    logger.info(
        f"[agent] action={action} reason={decision.get('reason')} question={question[:50]}"
    )

    # 1. 需要澄清
    if action == "clarify":
        return {
            "status": "NEED_CLARIFICATION",
            "answer": "",
            "references": [],
            "confidence": 0.0,
            "need_clarification": True,
            "clarification_question": decision.get("clarification_question", ""),
            "reason": decision.get("reason", "问题需要澄清")
        }

    # 2. 直接回答
    if action == "direct_answer":
        return {
            "status": "OK",
            "answer": (
                "我是一个基于当前知识库的问答助手。"
                "我可以根据已加载文档回答问题，也可以列出知识库文档，"
                "并处理简单四则运算。对于知识库没有覆盖的问题，我会说明无法根据文档确定。"
            ),
            "references": [],
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "直接回答系统能力问题")
        }

    # 3. 拒答
    if action == "refuse":
        return {
            "status": "NO_RELEVANT_INFO",
            "answer": "这个问题不属于当前知识库覆盖范围，我目前只能根据已收录资料回答相关问题。",
            "references": [],
            "confidence": 0.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": decision.get("reason", "问题超出知识库范围")
        }

    # 4. 调用 list_documents 工具
    if action == "list_documents":
        tool_result = dispatch_tool(action, decision.get("tool_args", {}))

        if tool_result.get("error"):
            return {
                "status": "MODEL_BAD_OUTPUT",
                "answer": "",
                "references": [],
                "confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": tool_result.get("error", "list_documents 工具执行失败")
            }

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
    # 5. 调用 simple_calculator 工具
    if action == "simple_calculator":
        tool_result = dispatch_tool("simple_calculator", decision["tool_args"])

        if not tool_result.get("ok"):
            return {
                "status": "MODEL_BAD_OUTPUT",
                "answer": "",
                "references": [],
                "confidence": 0.0,
                "need_clarification": False,
                "clarification_question": "",
                "reason": tool_result.get("error", "计算工具执行失败")
            }

        return {
            "status": "OK",
            "answer": f"计算结果是：{tool_result.get('result')}",
            "references": [],
            "confidence": 1.0,
            "need_clarification": False,
            "clarification_question": "",
            "reason": "Agent 选择 simple_calculator 工具"
        }

    # 6. 调用 search_knowledge_base，也就是原来的 RAG
    if action == "search_knowledge_base":
        logger.info(f"[retrieve] 开始检索 top_k={top_k}")
        retrieved_chunks = search_relevant_chunks(question, top_k)

        hit_brief = [
            {
                "chunk_id": chunk["chunk_id"],
                "score": round(chunk["hybrid_score"], 4),
                "bm25": round(chunk["bm25_score"], 4),
                "vector": round(chunk["vector_score"], 4),
                "file_name": chunk["file_name"],
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            f"[retrieve] 检索完成 chunks_len={len(retrieved_chunks)} hits={hit_brief}"
        )

        logger.info("[answer] 开始调用 RAG 问答逻辑")
        return ask(question, top_k=top_k, retrieved_chunks=retrieved_chunks)

    # 7. 防御分支：理论上不应该走到这里
    return {
        "status": "MODEL_BAD_OUTPUT",
        "answer": "",
        "references": [],
        "confidence": 0.0,
        "need_clarification": False,
        "clarification_question": "",
        "reason": f"未知 Agent action: {action}"
    }

def is_out_of_scope(question:str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(keyword in q for keyword in OUT_OF_SCOPE_KEYWORDS)

# 简单计算判断函数
def extract_simple_math_expression(question:str) -> str |None:
    q = (question or "").strip()

    # 去掉一些常见中文表达
    remove_words = [
        "是多少",
        "等于多少",
        "计算",
        "帮我算一下",
        "请帮我算一下",
        "请计算",
        "结果是多少",
        "？",
        "?",
        "，",
        ",",
        "。"
    ]

    for word in remove_words:
        q = q.replace(word, "")

    q = q.strip()

    # 只保留数字，小数点，四则运算符和括号
    allowed_chars = set("0123456789.+-*/() ")
    if q and all(ch in allowed_chars for ch in q):
        if any(op in q for op in ["+", "-", "*", "/"]):
            return q

    return None


# chat接口回复结构定义
def build_chat_response(code:int|str,
                        message:str,
                        question:str,
                        answer:str="",
                        references:list | None = None,
                        confidence: float = 0.0,
                        need_clarification: bool = False,
                        status: PublicStatus = "ok",
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
                            status=status,
                            clarification_question=clarification_question,
                            reason=reason
                            )
                        )

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


# =========================
# 接口暴露
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
    question = (req.question or "").strip()
    logger.info(f"[chat] 收到请求 question_len={len(question)}，preview={question[:50]}")
    try:
        if not question:
            logger.warning("[chat] final_status=clarify reason=用户未输入问题")
            response.status_code = 400
            return build_chat_response(
                code="EMPTY_QUESTION",
                message="问题不能为空，请输入你想查询的内容",
                question=question,
                confidence=0.0,
                need_clarification=True,
                status="clarify",
                clarification_question="请直接输入你想查询的具体问题，例如：年假申请流程是什么？",
                reason="用户未输入问题"
            )


        rag_status = get_rag_status()
        if not rag_status["ready"]:
            logger.warning("[chat] final_status=empty_index reason=知识库未准备完成")
            response.status_code = 503
            return build_chat_response(
                code="NO_DOCUMENTS",
                message=rag_status["message"],
                question=question,
                confidence=0.0,
                need_clarification=False,
                status="empty_index",
                clarification_question="",
                reason="知识库未准备完成"
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K

        try:
            logger.info("[answer] 开始处理：工具优先，否则走 RAG")
            result = run_agent_once(question, final_top_k)

        except Exception as e:
            logger.exception(f"[answer] 处理失败 error={e}")
            response.status_code = 500
            return build_chat_response(
                code="MODEL_ERROR",
                message="处理失败，请稍后重试",
                question=question,
                confidence=0.0,
                need_clarification=False,
                status="error",
                clarification_question="",
                reason="工具调用或模型调用异常"
            )

        if result["status"] == "NEED_CLARIFICATION":
            logger.warning(
                f"[chat] final_status=need_clarification reason=问题信息不足"
            )
            response.status_code = 200
            return build_chat_response(
                code="NEED_CLARIFICATION",
                message="问题信息不足，需要补充后再回答",
                question=question,
                answer="",
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=True,
                status="clarify",
                clarification_question=result.get("clarification_question", ""),
                reason=result.get("reason", "问题缺少关键限定信息")
            )

        if result["status"] == "NO_RELEVANT_INFO":
            logger.warning(
                f"[chat] final_status=no_evidence reason={result.get('reason', '检索证据不足')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="NO_RELEVANT_INFO",
                message="我没有在文档中找到相关信息",
                question=question,
                answer=result.get("answer", ""),
                references=result.get("references", []),
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                status="no_evidence",
                clarification_question="",
                reason=result.get("reason", "模型判断当前上下文不足以支撑回答")
            )

        if result["status"] == "MODEL_BAD_OUTPUT":
            logger.warning(
                f"[chat] final_status=error reason={result.get('reason', '模型输出格式异常')}"
            )
            response.status_code = 200
            return build_chat_response(
                code="MODEL_BAD_OUTPUT",
                message="模型输出格式异常，请稍候重试",
                question=question,
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False),
                status="error",
                clarification_question="",
                reason=result.get("reason", "模型返回结构不符合预期")
            )

        logger.info(
            f"[chat] final_status=ok reason={result.get('reason', '正常回答')}"
        )
        response.status_code = 200
        return build_chat_response(
            code=0,
            message="success",
            question=question,
            answer=result["answer"],
            references=result.get("references", []),
            confidence=result.get("confidence", 0.0),
            need_clarification=result.get("need_clarification", False),
            status="ok",
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





















