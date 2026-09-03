"""GitHub file payload 与 Phase 9 KnowledgeDocument 的纯转换。"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.ingestion import (
    KnowledgeDocument,
    KnowledgeSourceType,
    build_knowledge_document,
)


_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40,64}$")
_REPOSITORY_PART_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitHubFilePayload(BaseModel):
    """从受治理 GitHub MCP 结果中提取的有限文件载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    owner: StrictStr = Field(min_length=1, max_length=128)
    repository: StrictStr = Field(min_length=1, max_length=128)
    path: StrictStr = Field(min_length=1, max_length=1024)
    content: StrictStr = Field(repr=False)
    requested_ref: StrictStr | None = Field(default=None, max_length=256)
    resolved_sha: StrictStr | None = Field(default=None, max_length=64)
    mime_type: StrictStr | None = Field(default=None, max_length=255)

    @field_validator("owner", "repository")
    @classmethod
    def _normalize_repository_part(cls, value: str) -> str:
        normalized = value.strip().strip("/")
        if not _REPOSITORY_PART_PATTERN.fullmatch(normalized):
            raise ValueError("GitHub owner/repository 格式无效")
        return normalized

    @field_validator("path")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        normalized = value.strip().lstrip("/")
        if not normalized:
            raise ValueError("GitHub source path 不能为空")
        if any(part in {".", ".."} for part in normalized.split("/")):
            raise ValueError("GitHub source path 不允许相对路径段")
        return normalized

    @field_validator("requested_ref", "mime_type")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("resolved_sha")
    @classmethod
    def _validate_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _SHA_PATTERN.fullmatch(normalized):
            raise ValueError("GitHub resolved SHA 格式无效")
        return normalized


class GitHubFileSourceAdapter:
    """把已验证 GitHub file payload 纯转换为 KnowledgeDocument。"""

    def to_document(
        self,
        payload: GitHubFilePayload,
        *,
        fetched_at: datetime | None = None,
    ) -> KnowledgeDocument:
        """复用 Phase 9 factory 计算 canonical URI、identity 与安全正文。"""
        metadata = {
            "owner": payload.owner,
            "repository": payload.repository,
            "path": payload.path,
            "requested_ref": payload.requested_ref,
            "resolved_sha": payload.resolved_sha,
            "mime_type": payload.mime_type,
        }
        return build_knowledge_document(
            source_type=KnowledgeSourceType.GITHUB,
            source_uri=(
                f"github://{payload.owner}/{payload.repository}/{payload.path}"
            ),
            title=payload.path.rsplit("/", maxsplit=1)[-1],
            content=payload.content,
            fetched_at=fetched_at,
            version=payload.resolved_sha,
            mime_type=payload.mime_type,
            metadata={key: value for key, value in metadata.items() if value is not None},
            content_is_untrusted=True,
        )


__all__ = ["GitHubFilePayload", "GitHubFileSourceAdapter"]
