"""受治理外部知识来源的纯 Adapter 与显式 orchestration。"""

from app.sources.firecrawl import FirecrawlPagePayload, FirecrawlPageSourceAdapter
from app.sources.github import GitHubFilePayload, GitHubFileSourceAdapter
from app.sources.service import (
    ExternalKnowledgeSourceService,
    SourceIngestionError,
)


__all__ = [
    "ExternalKnowledgeSourceService",
    "FirecrawlPagePayload",
    "FirecrawlPageSourceAdapter",
    "GitHubFilePayload",
    "GitHubFileSourceAdapter",
    "SourceIngestionError",
]
