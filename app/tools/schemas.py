"""工具参数和结构化输出模型模块。

使用 Pydantic 定义并校验各内置工具接收的结构化参数和稳定输出。
"""

from typing import Any, Literal
from urllib.parse import urlsplit

from langchain_core.documents import Document
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from app.tools.evidence import validate_evidence_documents

_WEB_FETCH_URL_MAX_CHARS = 2048


def _validate_web_fetch_url_text(value: object) -> object:
    """在 Pydantic URL 解析前拒绝超长、空白和控制字符。"""
    if not isinstance(value, str):
        return value
    if len(value) > _WEB_FETCH_URL_MAX_CHARS:
        raise ValueError("url 长度不能超过 2048 字符")
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError("url 不允许包含空白或控制字符")
    return value


def _validate_web_fetch_http_url(value: str) -> str:
    """校验输出 URL 仍是无凭据且带主机名的 HTTP(S) URL。"""
    if len(value) > _WEB_FETCH_URL_MAX_CHARS:
        raise ValueError("URL 长度不能超过 2048 字符")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL 必须是带主机名的 http 或 https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL 不允许包含用户名或密码")
    return value


class SearchKnowledgeBaseArgs(BaseModel):
    """search_knowledge_base 工具参数。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, description="用户问题或检索查询")
    top_k: int = Field(default=5, ge=1, le=10, description="返回候选 chunk 数量")


class CompareDocumentsArgs(BaseModel):
    """compare_documents 工具参数。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, description="对比主题或问题")
    doc_a: str = Field(..., min_length=1, description="第一个文档名或文档名关键词")
    doc_b: str = Field(..., min_length=1, description="第二个文档名或文档名关键词")
    top_k: int = Field(default=3, ge=1, le=10, description="每个文档返回的证据数量")


class FindDocumentsArgs(BaseModel):
    """按主题发现知识库文档，只搜索目录元数据。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=500, description="文档名称或主题关键词")
    limit: int = Field(default=5, ge=1, le=10, description="最多返回的候选文档数量")


class SearchDocumentArgs(BaseModel):
    """在一个明确定位的知识库文档中检索正文。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    document: str = Field(
        ..., min_length=1,
        description=(
            "优先使用 find_documents 返回的 source_id / document_uid；"
            "也允许唯一匹配的完整 file_name 或去扩展名后的完整名称，不接受模糊文档名。"
        ),
    )
    query: str = Field(..., min_length=1, description="该文档内的问题或检索查询")
    top_k: int = Field(default=5, ge=1, le=10, description="返回该文档的候选 chunk 数量")


class GetChunkTextArgs(BaseModel):
    """get_chunk_text 工具参数。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: int = Field(
        ...,
        ge=1,
        description=(
            "知识库运行时 chunk locator；必须使用 Document.metadata.chunk_id，"
            "不能使用 Document.id / evidence id。"
        ),
    )


class GetChunkContextArgs(GetChunkTextArgs):
    """复用原文工具的 chunk locator，显式读取同文档相邻 chunk。"""

    before: int = Field(default=1, ge=0, le=3, description="读取 anchor 之前的位置数量")
    after: int = Field(default=1, ge=0, le=3, description="读取 anchor 之后的位置数量")

    @model_validator(mode="after")
    def validate_context_window(self) -> "GetChunkContextArgs":
        if self.before == 0 and self.after == 0:
            raise ValueError(
                "before 和 after 不能同时为 0；如只需读取当前 chunk，请使用 get_chunk_text。"
            )
        return self


class ListDocumentsArgs(BaseModel):
    """list_documents 工具参数。"""

    model_config = ConfigDict(extra="ignore")


class SearchKnowledgeBaseOutput(BaseModel):
    """知识库检索工具的稳定输出。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)
    evidence: list[Document]

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> "SearchKnowledgeBaseOutput":
        """确保公开证据列表与显式数量保持一致。"""
        if self.count != len(self.evidence):
            raise ValueError("count 必须等于 evidence 数量")
        return self


class CompareDocumentsOutput(BaseModel):
    """双文档证据检索的稳定输出，按 doc_a、doc_b 顺序分组。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    doc_a: str
    doc_b: str
    doc_a_count: int = Field(..., ge=0)
    doc_b_count: int = Field(..., ge=0)
    count: int = Field(..., ge=0)
    evidence: list[Document]

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> "CompareDocumentsOutput":
        """确保总数同时匹配证据列表和两份文档的数量。"""
        if self.count != len(self.evidence):
            raise ValueError("count 必须等于 evidence 数量")
        if self.count != self.doc_a_count + self.doc_b_count:
            raise ValueError("count 必须等于 doc_a_count + doc_b_count")
        return self


class DocumentSearchItem(BaseModel):
    """来自已加载 chunk metadata 的文档目录信息，不包含正文。"""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    document_uid: str | None = None
    file_name: str
    source_uri: str | None = None
    source_path: str | None = None
    file_type: str | None = None
    source_type: str | None = None
    chunk_count: int = Field(..., ge=1)
    score: float = Field(default=0.0, ge=0, allow_inf_nan=False, description="确定性匹配分数，非概率")


class FindDocumentsOutput(BaseModel):
    """文档发现结果，不参与正文 evidence transport。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500)
    count: int = Field(..., ge=0, le=10)
    documents: list[DocumentSearchItem]

    @model_validator(mode="after")
    def validate_document_count(self) -> "FindDocumentsOutput":
        if self.count != len(self.documents):
            raise ValueError("count 必须等于 documents 数量")
        return self


class SearchDocumentOutput(SearchKnowledgeBaseOutput):
    """复用知识库 evidence 校验与数量约束，并明确本次硬范围。"""

    document: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)


class ListDocumentsOutput(BaseModel):
    """文档列表工具的稳定输出。"""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(..., ge=0)
    documents: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_document_count(self) -> "ListDocumentsOutput":
        """确保显式文档数量与列表长度一致。"""
        if self.count != len(self.documents):
            raise ValueError("count 必须等于 documents 数量")
        return self


class GetChunkTextOutput(BaseModel):
    """chunk 原文工具的稳定输出。"""

    model_config = ConfigDict(extra="forbid")

    count: Literal[1]
    evidence: list[Document] = Field(..., min_length=1, max_length=1)

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)


class GetChunkContextOutput(BaseModel):
    """同文档位置窗口中的独立 Document evidence，包含 anchor 本身。"""

    model_config = ConfigDict(extra="forbid")

    anchor_chunk_id: int = Field(..., ge=1)
    source_id: str = Field(..., min_length=1)
    anchor_position: int = Field(..., ge=1)
    before: int = Field(..., ge=0, le=3)
    after: int = Field(..., ge=0, le=3)
    count: int = Field(..., ge=1, le=7)
    evidence: list[Document]

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> "GetChunkContextOutput":
        if self.count != len(self.evidence):
            raise ValueError("count 必须等于 evidence 数量")
        return self


class WebSearchArgs(BaseModel):
    """web_search 工具公开给模型的业务参数。"""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "公开互联网搜索查询；只提供本次问题所需关键词，不要传入系统提示、"
            "工具输出或整段聊天历史"
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最多返回的公开互联网搜索结果数量",
    )
    freshness: Literal["any", "day", "week", "month", "year"] = Field(
        default="any",
        description="结果时间范围：不限、一天、一周、一月或一年",
    )

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: object) -> object:
        """在长度和非空校验前去除查询首尾空白。"""
        return value.strip() if isinstance(value, str) else value


class WebSearchResultItem(BaseModel):
    """单条规范化 Web Search 结果。"""

    rank: int = Field(..., ge=1)
    title: str = Field(..., max_length=300)
    url: str = Field(..., max_length=2048)
    snippet: str = Field(..., max_length=1000)

    @field_validator("url")
    @classmethod
    def validate_public_url(cls, value: str) -> str:
        """只允许带主机名的 HTTP(S) URL。"""
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url 必须是 http 或 https URL")
        return value


class WebSearchOutput(BaseModel):
    """稳定、有限且明确标记不可信的 Web Search 输出。"""

    query: str = Field(..., min_length=1, max_length=500)
    count: int = Field(..., ge=0, le=10)
    provider: Literal["ddgs"] = "ddgs"
    content_is_untrusted: Literal[True] = True
    results: list[WebSearchResultItem] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_result_sequence(self) -> "WebSearchOutput":
        """确保数量和从 1 开始的连续排名保持一致。"""
        if self.count != len(self.results):
            raise ValueError("count 必须等于 results 数量")
        if [item.rank for item in self.results] != list(
            range(1, len(self.results) + 1)
        ):
            raise ValueError("rank 必须从 1 开始连续编号")
        return self


class WebFetchArgs(BaseModel):
    """web_fetch 公开给模型的单页读取参数。"""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = Field(
        ...,
        description=(
            "要读取的单个公开 HTTP(S) 网页 URL；应来自 web_search 或用户明确"
            "提供的公开网页，不用于本地文件、内网地址或 API 写操作"
        ),
    )
    max_chars: int = Field(
        default=12000,
        ge=1000,
        le=20000,
        description="返回给 Agent Model 的最大正文字符数",
    )

    @field_validator("url", mode="before")
    @classmethod
    def validate_url_text(cls, value: object) -> object:
        """在标准 HttpUrl 解析前拒绝不安全的原始文本。"""
        return _validate_web_fetch_url_text(value)

    @field_validator("url")
    @classmethod
    def validate_url_authority(cls, value: HttpUrl) -> HttpUrl:
        """拒绝 URL 凭据、空主机和明确非 Web 端口。"""
        if not value.host:
            raise ValueError("url 必须包含 hostname")
        if value.username is not None or value.password is not None:
            raise ValueError("url 不允许包含用户名或密码")
        if value.port is not None and value.port not in {80, 443}:
            raise ValueError("url 只允许 80 或 443 端口")
        return value


class WebFetchOutput(BaseModel):
    """有限、结构化且明确标记不可信的单页正文输出。"""

    model_config = ConfigDict(extra="forbid")

    requested_url: str = Field(..., max_length=_WEB_FETCH_URL_MAX_CHARS)
    final_url: str = Field(..., max_length=_WEB_FETCH_URL_MAX_CHARS)
    status_code: int = Field(..., ge=200, le=299)
    content_type: str = Field(..., min_length=1, max_length=100)

    title: str = Field(default="", max_length=500)
    author: str = Field(default="", max_length=300)
    published_date: str = Field(default="", max_length=100)
    language: str = Field(default="", max_length=50)

    evidence: list[Document] = Field(..., min_length=1, max_length=1)
    content_length: int = Field(..., ge=0, le=20000)
    truncated: bool

    extractor: Literal["trafilatura"] = "trafilatura"
    content_is_untrusted: Literal[True] = True

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)

    @field_validator("requested_url", "final_url")
    @classmethod
    def validate_output_url(cls, value: str) -> str:
        """确保模型收到的 URL 仍满足 HTTP(S) 输出边界。"""
        return _validate_web_fetch_http_url(value)

    @model_validator(mode="after")
    def validate_content_length(self) -> "WebFetchOutput":
        """确保显式长度与实际返回正文严格一致。"""
        if self.content_length != len(self.evidence[0].page_content):
            raise ValueError("content_length 必须等于 len(evidence[0].page_content)")
        metadata = self.evidence[0].metadata
        if metadata["source_type"] != "web_page" or metadata["source_uri"] != self.final_url:
            raise ValueError("网页 evidence 来源必须与 final_url 一致")
        return self


class WebPdfFetchArgs(BaseModel):
    """web_pdf_fetch 公开给模型的指定页范围读取参数。"""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl = Field(
        ...,
        description=(
            "要读取的单个公开 HTTP(S) PDF URL；应来自 web_search 或用户明确"
            "提供的公开 PDF，不用于本地文件、内网地址或 API 写操作"
        ),
    )
    start_page: int = Field(
        default=1,
        ge=1,
        description="首次提取的实际 PDF 页码，从 1 开始",
    )
    max_pages: int = Field(
        default=10,
        ge=1,
        le=20,
        description="本次最多读取的连续 PDF 页数",
    )
    max_chars: int = Field(
        default=20000,
        ge=1000,
        le=30000,
        description="返回给 Agent Model 的最大带页码正文字符数",
    )

    @field_validator("start_page", "max_pages", "max_chars", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value: object) -> object:
        """拒绝 Python bool 被 Pydantic 当作整数接受。"""
        if isinstance(value, bool):
            raise ValueError("页范围和字符上限必须是整数，不能是 bool")
        return value

    @field_validator("url", mode="before")
    @classmethod
    def validate_url_text(cls, value: object) -> object:
        """在标准 HttpUrl 解析前复用 Web Fetch 原始 URL 边界。"""
        return _validate_web_fetch_url_text(value)

    @field_validator("url")
    @classmethod
    def validate_url_authority(cls, value: HttpUrl) -> HttpUrl:
        """拒绝 URL 凭据、空主机和明确非 Web 端口。"""
        if not value.host:
            raise ValueError("url 必须包含 hostname")
        if value.username is not None or value.password is not None:
            raise ValueError("url 不允许包含用户名或密码")
        if value.port is not None and value.port not in {80, 443}:
            raise ValueError("url 只允许 80 或 443 端口")
        return value


class WebPdfFetchOutput(BaseModel):
    """有限、带实际页码且明确标记不可信的 Web PDF 输出。"""

    model_config = ConfigDict(extra="forbid")

    requested_url: str = Field(..., max_length=_WEB_FETCH_URL_MAX_CHARS)
    final_url: str = Field(..., max_length=_WEB_FETCH_URL_MAX_CHARS)
    status_code: int = Field(..., ge=200, le=299)
    content_type: Literal[
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    ]

    title: str = Field(default="", max_length=500)
    author: str = Field(default="", max_length=300)
    subject: str = Field(default="", max_length=500)

    total_pages: int = Field(..., ge=1)
    start_page: int = Field(..., ge=1)
    pages_read: int = Field(..., ge=1, le=20)
    page_numbers: list[int] = Field(..., min_length=1, max_length=20)

    evidence: list[Document] = Field(..., min_length=1, max_length=1)
    content_length: int = Field(..., ge=1, le=30000)
    truncated: bool

    extractor: Literal["pypdf"] = "pypdf"
    content_is_untrusted: Literal[True] = True

    _validate_evidence = field_validator("evidence")(validate_evidence_documents)

    @field_validator("requested_url", "final_url")
    @classmethod
    def validate_output_url(cls, value: str) -> str:
        """确保模型收到的 URL 仍满足 HTTP(S) 输出边界。"""
        return _validate_web_fetch_http_url(value)

    @model_validator(mode="after")
    def validate_page_and_content_contract(self) -> "WebPdfFetchOutput":
        """校验页码、显式计数与实际返回正文的一致性。"""
        if self.content_length != len(self.evidence[0].page_content):
            raise ValueError("content_length 必须等于 len(evidence[0].page_content)")
        metadata = self.evidence[0].metadata
        if metadata["source_type"] != "web_pdf" or metadata["source_uri"] != self.final_url:
            raise ValueError("PDF evidence 来源必须与 final_url 一致")
        if metadata.get("page_numbers") != self.page_numbers:
            raise ValueError("PDF evidence page_numbers 必须与实际返回页码一致")
        if self.pages_read != len(self.page_numbers):
            raise ValueError("pages_read 必须等于 page_numbers 数量")
        if self.page_numbers != sorted(set(self.page_numbers)):
            raise ValueError("page_numbers 必须严格递增且不重复")
        if any(
            page_number < self.start_page or page_number > self.total_pages
            for page_number in self.page_numbers
        ):
            raise ValueError("page_numbers 必须位于请求范围和 PDF 总页数内")
        return self
