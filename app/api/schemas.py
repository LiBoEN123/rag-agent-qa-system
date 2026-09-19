"""API 数据模型模块。

使用 Pydantic 定义聊天、引用和检索接口的请求与响应字段约束。
"""

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.chat import PublicStatus, ToolMode


# 定义聊天任务与可选会话标识；检索数量由 Agent Tool Call 决定。
class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: str | None = Field(None, description="多轮会话 ID")
    tool_mode: ToolMode = ToolMode.FULL

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "普通本科四年制专业学生最长学习年限是多少？",
                "session_id": "demo_session_001",
            }
        }
    )


# 描述回答引用的文档片段、来源位置及检索评分。
class ChatReferences(BaseModel):
    reference_id: str
    file_name: str
    source_path: str
    chunk_id: str | int
    position: int | None = None
    snippet: str
    score: float
    rerank_score: float


# 封装回答、引用、置信度、处理状态及工具和模型调用信息。
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
    answer_llm_used: bool = False
    llm_call_count: int = Field(default=0, ge=0)


# 定义聊天接口的业务码、消息、追踪标识和回答数据结构。
class ChatResponse(BaseModel):
    code: int | str
    message: str
    trace_id: str = ""
    data: ChatData


class SearchRequest(BaseModel):
    """确定性诊断检索输入，独立于 Agent Chat 的请求合同。"""

    question: str = Field(..., description="检索问题")
    top_k: int | None = Field(None, ge=1, le=10, description="召回条数")


# 描述诊断检索命中的文档片段及各阶段评分。
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


# 封装检索问题、处理状态、原因和命中列表。
class SearchData(BaseModel):
    question: str
    status: PublicStatus
    reason: str = ""
    results: list[SearchResult] = Field(default_factory=list)


# 定义检索接口的业务码、消息和检索数据结构。
class SearchResponse(BaseModel):
    code: int | str
    message: str
    data: SearchData
