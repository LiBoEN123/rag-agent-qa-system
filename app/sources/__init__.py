"""受治理外部知识来源的纯 Adapter 与显式 orchestration。"""

from app.sources.firecrawl import FirecrawlPagePayload, FirecrawlPageSourceAdapter
from app.sources.github import GitHubFilePayload, GitHubFileSourceAdapter
from app.sources.service import (
    ExternalKnowledgeSourceService,
    FIRECRAWL_RECORD_NAMESPACE,
    GITHUB_RECORD_NAMESPACE,
    SourceIngestionError,
)


__all__ = [
    "ExternalKnowledgeSourceService",
    "FIRECRAWL_RECORD_NAMESPACE",
    "FirecrawlPagePayload",
    "FirecrawlPageSourceAdapter",
    "GITHUB_RECORD_NAMESPACE",
    "GitHubFilePayload",
    "GitHubFileSourceAdapter",
    "SourceIngestionError",
]
