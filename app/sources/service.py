"""受治理 MCP / 上传来源抓取、校验与显式 proposal。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from app.config import AppSettings, get_settings
from app.ingestion import (
    INGESTION_MAX_CONTENT_CHARS,
    IngestionCandidate,
    IngestionRequester,
    IngestionService,
)
from app.sources.firecrawl import FirecrawlPagePayload, FirecrawlPageSourceAdapter
from app.sources.github import GitHubFilePayload, GitHubFileSourceAdapter
from app.sources.upload import load_upload_document
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolRiskLevel
from app.tools.mcp.bridge import invoke_mcp_tool
from app.tools.mcp.provider import get_mcp_tool_provider
from app.tools.mcp.serialization import McpSerializationProfile


_GITHUB_TOOL = "github_get_file_contents"
_FIRECRAWL_TOOL = "firecrawl_firecrawl_scrape"
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40,64}$")


class SourceIngestionError(RuntimeError):
    """不携带正文或远端异常细节的 source orchestration 失败。"""

    # 保存来源入库错误码，并使用提供的消息初始化异常。
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# 校验 MCP 入库输出成功、正文完整且带有不可信标记和内容块列表。
def _require_ingestion_output(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if result.get("ok") is not True:
        raise SourceIngestionError("SOURCE_FETCH_FAILED", "受治理 MCP source fetch 失败")
    output = result.get("output")
    if not isinstance(output, dict):
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "MCP source payload 无效")
    if output.get("serialization_profile") != McpSerializationProfile.INGESTION.value:
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "source output profile 不匹配")
    if output.get("truncated") is not False:
        blocks = output.get("content_blocks")
        if isinstance(blocks, list) and any(
            isinstance(block, Mapping)
            and isinstance(block.get("size_chars"), int)
            and block["size_chars"] > INGESTION_MAX_CONTENT_CHARS
            for block in blocks
        ):
            raise SourceIngestionError("SOURCE_CONTENT_TOO_LARGE", "source 正文超过上限")
        raise SourceIngestionError("SOURCE_CONTENT_TRUNCATED", "source 正文不完整")
    if output.get("content_is_untrusted") is not True:
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "source trust marker 无效")
    if not isinstance(output.get("content_blocks"), list):
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "MCP content blocks 无效")
    return output


# 从内容块中匹配唯一的目标仓库文件，并提取正文及已解析的 SHA。
def _github_payload(
    output: Mapping[str, Any],
    *,
    owner: str,
    repository: str,
    path: str,
    requested_ref: str | None,
) -> GitHubFilePayload:
    requested_path = str(PurePosixPath(path.strip().lstrip("/")))
    matches: list[tuple[Mapping[str, Any], str | None]] = []
    for block in output["content_blocks"]:
        if not isinstance(block, Mapping) or block.get("type") != "embedded_resource_text":
            continue
        text = block.get("text")
        uri = block.get("uri")
        if not isinstance(text, str) or not isinstance(uri, str) or block.get("truncated"):
            continue
        parsed = urlsplit(uri)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if parsed.scheme != "repo" or (parsed.hostname or "").casefold() != owner.casefold():
            continue
        if not segments or segments[0].casefold() != repository.casefold():
            continue
        try:
            contents_index = segments.index("contents")
        except ValueError:
            continue
        if "/".join(segments[contents_index + 1 :]) != requested_path:
            continue
        sha = None
        if contents_index >= 3 and segments[1] == "sha" and _SHA_PATTERN.fullmatch(segments[2]):
            sha = segments[2].lower()
        matches.append((block, sha))
    if len(matches) != 1:
        raise SourceIngestionError(
            "SOURCE_PAYLOAD_INVALID",
            "GitHub source resource 缺失或不唯一",
        )
    block, resolved_sha = matches[0]
    return GitHubFilePayload(
        owner=owner,
        repository=repository,
        path=requested_path,
        requested_ref=requested_ref,
        resolved_sha=resolved_sha,
        content=block["text"],
        mime_type=block.get("mime_type"),
    )


# 仅在指定字段为字符串时返回其值，否则返回空值。
def _text_field(values: Mapping[str, Any], name: str) -> str | None:
    value = values.get(name)
    return value if isinstance(value, str) else None


# 从结构化内容或 JSON 文本块中提取网页正文及来源元数据。
def _firecrawl_payload(
    output: Mapping[str, Any],
    *,
    requested_url: str,
) -> FirecrawlPagePayload:
    payloads: list[Mapping[str, Any]] = []
    structured = output.get("structured_content")
    if isinstance(structured, Mapping):
        payloads.append(structured)
    for block in output["content_blocks"]:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = block.get("text")
        if not isinstance(text, str) or block.get("truncated"):
            continue
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, Mapping):
            payloads.append(decoded)

    selected: Mapping[str, Any] | None = None
    content: str | None = None
    for candidate in payloads:
        for key in ("markdown", "content", "text"):
            value = candidate.get(key)
            if isinstance(value, str):
                selected, content = candidate, value
                break
        if selected is not None:
            break
    if selected is None or content is None:
        raise SourceIngestionError(
            "SOURCE_PAYLOAD_INVALID",
            "Firecrawl scrape 未返回可证明完整的正文",
        )
    if (
        selected.get("success", True) is not True
        or selected.get("truncated", False) is not False
    ):
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "Firecrawl page 失败或不完整")
    metadata = selected.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    version = _text_field(metadata, "etag") or _text_field(metadata, "lastModified")
    status_code = metadata.get("statusCode")
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None
    if status_code is not None and not 200 <= status_code < 300:
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "Firecrawl page HTTP 状态无效")
    html = selected.get("html")
    if html is not None and not isinstance(html, str):
        raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "Firecrawl HTML 字段无效")
    return FirecrawlPagePayload(
        requested_url=requested_url,
        content=content,
        html=html,
        title=_text_field(metadata, "title"),
        final_url=_text_field(metadata, "url"),
        mime_type=_text_field(metadata, "contentType"),
        language=_text_field(metadata, "language"),
        version=version,
        status_code=status_code,
    )


class ExternalKnowledgeSourceService:
    """显式 operator/service source fetch 与 proposal。"""

    # 配置申请主体、入库服务和来源适配器。
    def __init__(
        self,
        *,
        requester: IngestionRequester,
        ingestion_service: IngestionService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._requester = requester
        self._ingestion = ingestion_service or IngestionService()
        self._settings = settings
        self._github_adapter = GitHubFileSourceAdapter()
        self._firecrawl_adapter = FirecrawlPageSourceAdapter()

    # 返回当前来源服务使用的入库领域服务。
    @property
    def ingestion_service(self) -> IngestionService:
        return self._ingestion

    # 查找调用者可见的 MCP 来源工具，并要求其声明为只读工具。
    def _tool(self, name: str, context: ToolInvocationContext):
        provider = get_mcp_tool_provider()
        if provider is None:
            raise SourceIngestionError("SOURCE_TOOL_UNAVAILABLE", "MCP runtime 未初始化")
        tools = {tool.name: tool for tool in provider.list_tools(context=context)}
        tool = tools.get(name)
        if tool is None:
            raise SourceIngestionError("SOURCE_TOOL_UNAVAILABLE", "source MCP tool 不可用")
        if (tool.metadata or {}).get("risk_level") != ToolRiskLevel.READ_ONLY.value:
            raise SourceIngestionError("SOURCE_TOOL_NOT_READ_ONLY", "source MCP tool 非只读")
        return tool

    # 通过受治理的 MCP 入口抓取内容，并校验其满足完整入库输出要求。
    async def _fetch(
        self,
        name: str,
        args: dict[str, Any],
        context: ToolInvocationContext,
    ) -> dict[str, Any]:
        result = await invoke_mcp_tool(
            self._tool(name, context),
            args,
            context=context,
            serialization_profile=McpSerializationProfile.INGESTION,
        )
        return _require_ingestion_output(result)

    # 以配置的申请主体和申请原因提交知识文档入库提案。
    def _propose(self, document, reason: str | None) -> IngestionCandidate:
        return self._ingestion.propose(
            document=document,
            requester=self._requester,
            reason=reason,
        )

    # 抓取并适配 GitHub 文件，将生成的知识文档提交为入库候选。
    async def propose_github_file(
        self,
        *,
        owner: str,
        repository: str,
        path: str,
        context: ToolInvocationContext,
        ref: str | None = None,
        fetched_at: datetime | None = None,
        reason: str | None = None,
    ) -> IngestionCandidate:
        args = {"owner": owner, "repo": repository, "path": path}
        if ref:
            args["ref"] = ref
        output = await self._fetch(_GITHUB_TOOL, args, context)
        payload = _github_payload(
            output,
            owner=owner,
            repository=repository,
            path=path,
            requested_ref=ref,
        )
        document = self._github_adapter.to_document(payload, fetched_at=fetched_at)
        return self._propose(document, reason)

    # 抓取网页主体 Markdown 和 cleaned HTML，复用结构 parser 后提交入库提案。
    async def propose_firecrawl_page(
        self,
        *,
        url: str,
        context: ToolInvocationContext,
        fetched_at: datetime | None = None,
        reason: str | None = None,
    ) -> IngestionCandidate:
        output = await self._fetch(
            _FIRECRAWL_TOOL,
            {
                "url": url,
                "formats": ["markdown", "html"],
                "onlyMainContent": True,
                "removeBase64Images": True,
            },
            context,
        )
        payload = _firecrawl_payload(output, requested_url=url)
        try:
            document = self._firecrawl_adapter.to_document(payload, fetched_at=fetched_at)
        except ValueError:
            raise SourceIngestionError("SOURCE_PAYLOAD_INVALID", "Firecrawl page 无法转换为知识文档") from None
        return self._propose(document, reason)

    def propose_upload_file(
        self, *, file: BinaryIO, filename: str | None, reason: str | None = None,
    ) -> IngestionCandidate:
        """解析上传全文并复用候选治理；文件由 HTTP adapter 管理关闭。"""
        settings = self._settings or get_settings()
        document = load_upload_document(
            file, filename=filename, max_bytes=settings.ingestion_upload_max_bytes,
        )
        return self._propose(document, reason)
