"""API 数据模型模块。

使用 Pydantic 定义聊天、引用和检索接口的请求与响应字段约束。
"""

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.chat import PublicStatus


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int | None = Field(None, ge=1, le=10, description="召回条数")
    decompose_top_k: int | None = Field(
        None, ge=1, le=10, description="问题拆分后的召回条数"
    )
    session_id: str | None = Field(None, description="多轮会话 ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "普通本科四年制专业学生最长学习年限是多少？",
                "top_k": 5,
                "session_id": "demo_session_001",
            }
        }
    )


class ChatReferences(BaseModel):
    reference_id: str
    file_name: str
    source_path: str
    chunk_id: str | int
    position: int | None = None
    snippet: str
    score: float
    rerank_score: float


class ChatData(BaseModel):
    question: str
    answer: str
    references: list[ChatReferences] = Field(default_factory=list)
    retrieval_confidence: float = Field(..., ge=0.0, le=1.0)
    answer_confidence: float = Field(..., ge=0.0, le=1.0)
    need_clarification: bool
    status: PublicStatus
    clarification_question: str
    reason: str
    action: str = ""
    tool_result_summary: str = ""
    tool_trace: list[dict] = Field(default_factory=list)
    router_llm_used: bool = False
    decompose_llm_used: bool = False
    answer_llm_used: bool = False
    llm_call_count: int = Field(default=0, ge=0)


class ChatResponse(BaseModel):
    code: int | str
    message: str
    trace_id: str = ""
    data: ChatData


class SearchResult(BaseModel):
    reference_id: str
    source_path: str
    file_name: str
    chunk_id: str | int
    position: int | None = None
    snippet: str
    score: float
    bm25_score: float
    vector_score: float
    rerank_score: float


class SearchData(BaseModel):
    question: str
    status: PublicStatus
    reason: str = ""
    results: list[SearchResult] = Field(default_factory=list)


class SearchResponse(BaseModel):
    code: int | str
    message: str
    data: SearchData
