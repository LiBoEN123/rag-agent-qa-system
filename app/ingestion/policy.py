"""无副作用、无网络、无 LLM 的 Knowledge ingestion policy。"""

from __future__ import annotations

from collections.abc import Collection

from app.ingestion.contracts import (
    IngestionPolicyDecision,
    KnowledgeDocument,
    KnowledgeSourceType,
    PolicyDecisionType,
    PolicyReason,
    TrustLevel,
)


INGESTION_MAX_CONTENT_CHARS = 1_000_000
DEFAULT_SUPPORTED_SOURCE_TYPES = frozenset(KnowledgeSourceType)
_EXTERNAL_SOURCE_TYPES = frozenset(
    {
        KnowledgeSourceType.UPLOAD,
        KnowledgeSourceType.GITHUB,
        KnowledgeSourceType.WEB,
        KnowledgeSourceType.FIRECRAWL,
    }
)


def evaluate_ingestion_candidate(
    document: KnowledgeDocument,
    *,
    max_content_chars: int = INGESTION_MAX_CONTENT_CHARS,
    supported_source_types: Collection[KnowledgeSourceType] | None = None,
) -> IngestionPolicyDecision:
    """按显式 source/trust/size 规则评估文档，绝不产生副作用。"""
    if (
        isinstance(max_content_chars, bool)
        or not isinstance(max_content_chars, int)
        or max_content_chars < 1
    ):
        raise ValueError("max_content_chars 必须是正整数")
    supported = (
        DEFAULT_SUPPORTED_SOURCE_TYPES
        if supported_source_types is None
        else frozenset(supported_source_types)
    )
    if document.source_type not in supported:
        return IngestionPolicyDecision(
            decision=PolicyDecisionType.DENY,
            reasons=(PolicyReason.UNSUPPORTED_SOURCE,),
        )
    if not document.content.strip():
        return IngestionPolicyDecision(
            decision=PolicyDecisionType.DENY,
            reasons=(PolicyReason.EMPTY_CONTENT,),
        )
    if len(document.content) > max_content_chars:
        return IngestionPolicyDecision(
            decision=PolicyDecisionType.DENY,
            reasons=(PolicyReason.CONTENT_TOO_LARGE,),
        )

    if document.source_type is KnowledgeSourceType.LOCAL_FILE:
        if document.trust_level is TrustLevel.TRUSTED_INTERNAL:
            return IngestionPolicyDecision(
                decision=PolicyDecisionType.ALLOW,
                reasons=(PolicyReason.TRUSTED_LOCAL,),
            )
        return IngestionPolicyDecision(
            decision=PolicyDecisionType.REQUIRE_APPROVAL,
            reasons=(PolicyReason.SOURCE_REQUIRES_APPROVAL,),
        )

    if document.source_type in _EXTERNAL_SOURCE_TYPES:
        if document.trust_level is TrustLevel.TRUSTED_INTERNAL:
            return IngestionPolicyDecision(
                decision=PolicyDecisionType.DENY,
                reasons=(PolicyReason.INVALID_TRUST_ASSIGNMENT,),
            )
        if document.trust_level is TrustLevel.TRUSTED_EXTERNAL:
            return IngestionPolicyDecision(
                decision=PolicyDecisionType.ALLOW,
                reasons=(PolicyReason.TRUSTED_EXTERNAL,),
            )
        return IngestionPolicyDecision(
            decision=PolicyDecisionType.REQUIRE_APPROVAL,
            reasons=(
                PolicyReason.UNTRUSTED_EXTERNAL,
                PolicyReason.SOURCE_REQUIRES_APPROVAL,
            ),
        )

    return IngestionPolicyDecision(
        decision=PolicyDecisionType.DENY,
        reasons=(PolicyReason.UNSUPPORTED_SOURCE,),
    )


__all__ = [
    "DEFAULT_SUPPORTED_SOURCE_TYPES",
    "INGESTION_MAX_CONTENT_CHARS",
    "evaluate_ingestion_candidate",
]
