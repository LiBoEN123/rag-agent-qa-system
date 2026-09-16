"""统一构造 KnowledgeDocument，避免 source adapter 重复实现 identity。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.ingestion.contracts import (
    KnowledgeDocument,
    KnowledgeSourceType,
    TrustLevel,
)
from app.ingestion.document_blocks import ParsedDocument, canonical_document_text
from app.ingestion.identity import (
    build_content_hash,
    build_document_uid,
    build_source_id,
    build_structure_hash,
    canonicalize_source_uri,
)
from app.ingestion.normalization import prepare_content, prepare_metadata, prepare_structured_content


def default_trust_level(source_type: KnowledgeSourceType) -> TrustLevel:
    """返回 source adapter 未显式提供部署信任判断时的 fail-closed 默认值。"""
    if source_type is KnowledgeSourceType.LOCAL_FILE:
        return TrustLevel.TRUSTED_INTERNAL
    return TrustLevel.UNTRUSTED_EXTERNAL


def build_knowledge_document(
    *,
    source_type: KnowledgeSourceType | str,
    source_uri: str,
    title: str,
    content: str,
    structured_content: ParsedDocument | None = None,
    fetched_at: datetime | None = None,
    version: str | None = None,
    mime_type: str | None = None,
    language: str | None = None,
    metadata: dict[str, Any] | None = None,
    trust_level: TrustLevel | str | None = None,
    content_is_untrusted: bool = False,
) -> KnowledgeDocument:
    """构造脱敏、可验证文档；有结构时以其 canonical 视图生成 content。"""
    resolved_source_type = KnowledgeSourceType(source_type)
    canonical_uri = canonicalize_source_uri(resolved_source_type, source_uri)
    structure, structure_redacted = prepare_structured_content(structured_content)
    normalized_content, content_redacted = prepare_content(
        canonical_document_text(structure) if structure is not None else content
    )
    safe_metadata, metadata_redacted = prepare_metadata(
        metadata if metadata is not None else {}
    )
    source_id = build_source_id(resolved_source_type, canonical_uri)

    resolved_trust = (
        default_trust_level(resolved_source_type)
        if trust_level is None
        else TrustLevel(trust_level)
    )
    if type(content_is_untrusted) is not bool:
        raise TypeError("content_is_untrusted 必须是 bool")
    if content_is_untrusted:
        resolved_trust = TrustLevel.UNTRUSTED_EXTERNAL

    return KnowledgeDocument(
        source_id=source_id,
        document_uid=build_document_uid(source_id),
        source_type=resolved_source_type,
        source_uri=canonical_uri,
        title=title,
        content=normalized_content,
        content_hash=build_content_hash(normalized_content),
        structured_content=structure,
        structure_hash=build_structure_hash(structure) if structure is not None else None,
        content_redacted=content_redacted or structure_redacted,
        version=version,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        mime_type=mime_type,
        language=language,
        metadata=safe_metadata,
        metadata_redacted=metadata_redacted,
        trust_level=resolved_trust,
    )


__all__ = ["build_knowledge_document", "default_trust_level"]
