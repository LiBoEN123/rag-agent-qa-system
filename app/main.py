from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field, ConfigDict
from app.rag import init_rag, ask, preview_docs, preview_chunks, search_relevant_chunks, get_rag_status
from app.config import Settings
from app.untils import get_logger
import os
app = FastAPI(title='Mini Rag API', version='1.0.0')
logger = get_logger('chat_service')

class ChatRequest(BaseModel):
    question:str = Field(..., description='用户问题')
    top_k:int | None=Field(None, ge=1, le=10, description='召回条数')

    model_config = ConfigDict(
        json_schema_extra={
            "example":{
              "question": "购买七天后还能退款吗",
              "top_k": 5
            }
        }
    )

class ChatReferences(BaseModel):
    reference_id:str
    source_file:str
    chunk_id:str
    text:str

class ChatData(BaseModel):
    answer:str
    references:list[ChatReferences]=Field(default_factory=list)
    confidence:float = Field(..., ge=0.0, le=1.0, description="回答置信度，0~1")
    need_clarification:bool = Field(..., description="是否建议用户补充问题")

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
                "answer": "可以申请退款，但退款不由开发者操作，需联系苹果的支持团队处理。",
                "references": [
                  {
                    "reference_id": "Reference 1",
                    "source_file": "terms-of-service.zh.md",
                    "chunk_id": "terms-of-service.zh_chunk_5",
                    "text": "[标题路径] 4. 退款服务\n\n如果您在订阅赞助过程中遇到任何问题，都可以通过底部的邮件联系我们获得帮助。如果您仍然感到无法满足您的要求，可以选择申请退款。请注意，退款不由开发者操作，您需联系苹果的支持团队来处理退款事宜。  \n[申请退款](https://support.apple.com/118223)"
                  }
                ],
                "confidence":0.82,
                "need_clarification":False
              }
            }
        }
    )

class SearchResult(BaseModel):
    reference_id:str
    source_file:str
    chunk_id:str
    text:str
    score:float
    bm25_score:float
    vector_score:float

class SearchData(BaseModel):
    question:str
    references:list[SearchResult]=Field(default_factory=list)

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
                "question": "购买七天后还能退款吗",
                "references": [
                  {
                    "reference_id": "Reference 1",
                    "source_file": "terms-of-service.zh.md",
                    "chunk_id": "terms-of-service.zh_chunk_5",
                    "text": "[标题路径] 4. 退款服务\n\n如果您在订阅赞助过程中遇到任何问题，都可以通过底部的邮件联系我们获得帮助。如果您仍然感到无法满足您的要求，可以选择申请退款。请注意，退款不由开发者操作，您需联系苹果的支持团队来处理退款事宜。  \n[申请退款](https://support.apple.com/118223)",
                    "score": 1,
                    "bm25_score": 7.186296820637358,
                    "vector_score": 0.3061803877353668
                  },
                  {
                    "reference_id": "Reference 2",
                    "source_file": "terms-of-service.zh.md",
                    "chunk_id": "terms-of-service.zh_chunk_4",
                    "text": "[标题路径] 3. 订阅赞助\n\n如果您决定订阅赞助，那么您对我的付出表示感谢，就像偶尔给我买杯咖啡一样。捐赠后将不再弹出赞助提示，并解锁部分实用功能。感谢您的慷慨支持，我将信心十足地继续维护 DevHub 并确保其健康运行。",
                    "score": 0.7092396020889282,
                    "bm25_score": 4.213482890118754,
                    "vector_score": 0.21269017457962036
                  },
                  {
                    "reference_id": "Reference 3",
                    "source_file": "terms-of-service.zh.md",
                    "chunk_id": "terms-of-service.zh_chunk_6",
                    "text": "[标题路径] 5. 使用条款的更改\n\n当有新的使用条款更新时，我将在此页面上进行更新，并立即生效。",
                    "score": 0.3499600887298584,
                    "bm25_score": 0,
                    "vector_score": 0.119601771235466
                  },
                  {
                    "reference_id": "Reference 4",
                    "source_file": "CHANGELOG.zh.md",
                    "chunk_id": "CHANGELOG.zh_chunk_11",
                    "text": "[标题路径] [v1.42.1](https://github.com/jaywcjlove/DevHub/releases/tag/v1.42.1)\n\n1. pref: 优化首页侧边栏展示性能\n2. ui: 更新首页\n3. ui: 更新关于页面\n4. ui: 优化生命倒计时布局",
                    "score": 0.3488190770149231,
                    "bm25_score": 0,
                    "vector_score": 0.11875034123659134
                  },
                  {
                    "reference_id": "Reference 5",
                    "source_file": "terms-of-service.zh.md",
                    "chunk_id": "terms-of-service.zh_chunk_1",
                    "text": "<p align=\"right\">\n<a href=\"./terms-of-service.md\">English</a>\n</p>\n<!--rehype:style=float: right; bottom: -36px; position: relative;-->  \n服务条款\n===  \n在订阅捐赠之前，您可以阅读应用的使用协议，以获取更多信息。",
                    "score": 0.32936036586761475,
                    "bm25_score": 0,
                    "vector_score": 0.1042303591966629
                  }
                ]
              }
            }
        }
    )

@app.on_event("startup")
def start_on_event():
    Settings.validate()
    init_rag(
        str(Settings.RAG_FILE_PATH),
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP)

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
                        answer:str="",
                        references:list | None = None,
                        confidence: float = 0.0,
                        need_clarification: bool = False
                        ) -> ChatResponse:
    return ChatResponse(code=code,
                        message=message,
                        data=ChatData(
                            answer=answer,
                            references=references or [],
                            confidence=confidence,
                            need_clarification=need_clarification
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
    try:
        question = (req.question or "").strip()
        logger.info(f"[chat] 收到请求 question_len={len(question)}，preview={question[:50]}")
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
        print("runtime key suffix:", key[-4:] if key else None)

        if not question:
            logger.warning("[chat] 空问题请求")
            response.status_code = 400
            return build_chat_response(
                code="EMPTY_QUESTION",
                message="问题不能为空，请输入你想查询的内容",
                confidence=0.0,
                need_clarification=True
            )

        if is_out_of_scope(question):
            response.status_code = 503
            return build_chat_response(
                code="OUT_OF_SCOPE",
                message="这个问题不在当前知识库回答范围内",
                confidence=0.0,
                need_clarification=False
            )

        rag_status = get_rag_status()
        if not rag_status["ready"]:
            logger.warning("[chat] 数据目录为空")
            response.status_code = 503
            return build_chat_response(
                code="NO_DOCUMENTS",
                message=rag_status["message"],
                confidence=0.0,
                need_clarification=False
            )

        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K

        try:
            logger.info(f"[retrieve] 开始检索 top_k={final_top_k}")
            retrieved_chunks = search_relevant_chunks(question, final_top_k)
            hit_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
            logger.info(f"[retrieve] 检索完成 chunks_len={len(hit_ids)}，hit_ids={hit_ids}")

        except Exception as e:
            logger.exception(f"[retrieve] 检索失败 error={e}")
            response.status_code = 500
            return build_chat_response(
                code="RETRIEVE_ERROR",
                message="检索失败，请稍后重试",
                confidence=0.0,
                need_clarification=False
            )

        try:
            logger.info("[answer] 开始调用模型")
            result = ask(question, top_k=final_top_k, retrieved_chunks=retrieved_chunks)

        except Exception as e:
            logger.exception(f"[answer] 模型调用失败 error={e}")
            response.status_code = 500
            return build_chat_response(
                code="MODEL_ERROR",
                message="模型调用失败，请稍后重试",
                confidence=0.0,
                need_clarification=False
            )

        if result["status"] == "NO_RELEVANT_INFO":
            logger.warning("[chat] 检索证据不足，返回 honest fallback")
            response.status_code = 200
            return build_chat_response(
                code="NO_RELEVANT_INFO",
                message="我没有在文档中找到相关信息",
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False)
            )

        if result["status"] == "MODEL_BAD_OUTPUT":
            logger.warning("[chat] 模型输出格式异常")
            response.status_code = 200
            return build_chat_response(
                code="MODEL_BAD_OUTPUT",
                message="模型输出格式异常，请稍候重试",
                answer=result.get("answer", ""),
                references=[],
                confidence=result.get("confidence", 0.0),
                need_clarification=result.get("need_clarification", False)
            )

        logger.info("[chat] 请求处理完成")
        response.status_code = 200
        return build_chat_response(
            code=0,
            message="success",
            answer=result["answer"],
            references=result.get("references", []),
            confidence=result.get("confidence", 0.0),
            need_clarification=result.get("need_clarification", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[chat] 未知异常 error={e}")
        response.status_code = 500
        return build_chat_response(
            code="INTERNAL_ERROR",
            message="服务器内部错误",
            confidence=0.0,
            need_clarification=False
        )

@app.get("/docs-preview")
def docs_preview():
    docs = preview_docs()
    return {"count":len(docs), "data":docs}

@app.get("/chunks-preview")
def chunks_preview(
        limit:int = Query(20, ge=1, le=100),
        source_file:str | None=Query(None)):
    data = preview_chunks(limit, source_file)
    return {"count":len(data), "data":data}

@app.post("/search", response_model=SearchResponse)
def search_api(req:ChatRequest):
    try:
        final_top_k = req.top_k if req.top_k is not None else Settings.TOP_K
        retrieved_chunks = search_relevant_chunks(req.question, final_top_k)

        results = [
            SearchResult(
                reference_id=item["reference_id"],
                source_file=item["source_file"],
                chunk_id=item["chunk_id"],
                text=item["text"],
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
                references=results
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

















