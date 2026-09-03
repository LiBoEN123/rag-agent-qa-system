"""Firecrawl page payload 与 Phase 9 KnowledgeDocument 的纯转换。"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.ingestion import (
    KnowledgeDocument,
    KnowledgeSourceType,
    build_knowledge_document,
)
from app.ingestion.identity import canonicalize_source_uri


class FirecrawlPagePayload(BaseModel):
    """从受治理 Firecrawl scrape 结果中提取的有限页面载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    requested_url: StrictStr = Field(min_length=1, max_length=2048)
    content: StrictStr = Field(repr=False)
    title: StrictStr | None = Field(default=None, max_length=512)
    final_url: StrictStr | None = Field(default=None, max_length=2048)
    mime_type: StrictStr | None = Field(default=None, max_length=255)
    language: StrictStr | None = Field(default=None, max_length=64)
    version: StrictStr | None = Field(default=None, max_length=256)
    status_code: int | None = Field(default=None, ge=100, le=599)

    @field_validator("requested_url", "final_url")
    @classmethod
    def _canonicalize_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return canonicalize_source_uri(KnowledgeSourceType.FIRECRAWL, value)

    @field_validator("title", "mime_type", "language", "version")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class FirecrawlPageSourceAdapter:
    """把已验证 Firecrawl page payload 纯转换为 KnowledgeDocument。"""

    def to_document(
        self,
        payload: FirecrawlPagePayload,
        *,
        fetched_at: datetime | None = None,
    ) -> KnowledgeDocument:
        """以 canonical requested URL 固定 identity，redirect 仅进 metadata。"""
        title = payload.title
        if title is None:
            title = urlsplit(payload.requested_url).hostname or "Firecrawl page"
        metadata = {
            "final_url": payload.final_url,
            "status_code": payload.status_code,
            "mime_type": payload.mime_type,
            "language": payload.language,
        }
        return build_knowledge_document(
            source_type=KnowledgeSourceType.FIRECRAWL,
            source_uri=payload.requested_url,
            title=title,
            content=payload.content,
            fetched_at=fetched_at,
            version=payload.version,
            mime_type=payload.mime_type,
            language=payload.language,
            metadata={key: value for key, value in metadata.items() if value is not None},
            content_is_untrusted=True,
        )


__all__ = ["FirecrawlPagePayload", "FirecrawlPageSourceAdapter"]
