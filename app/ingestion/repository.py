"""Phase 9 的最小候选 repository contract 与内存实现。"""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from app.ingestion.contracts import IngestionCandidate, IngestionCandidateStatus, KnowledgeSourceType


class CandidateRepository(Protocol):
    """候选存储边界；基础设施层可以提供持久实现。"""

    def get(self, candidate_id: str) -> IngestionCandidate | None:
        """按稳定 candidate ID 查询。"""

    def save(self, candidate: IngestionCandidate) -> IngestionCandidate:
        """保存或幂等返回完全相同的候选。"""

    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        """在 Service 已验证状态迁移后替换现有候选。"""

    def count(self) -> int:
        """返回逻辑候选数量。"""

    def list(
        self,
        *,
        status: IngestionCandidateStatus | None = None,
        source_type: KnowledgeSourceType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionCandidate]:
        """按可查询字段分页返回候选，不改变 domain state。"""


class InMemoryCandidateRepository:
    """线程安全且不会为同一 candidate_id 建立逻辑重复的内存存储。"""

    # 初始化候选内存字典和保护读写操作的可重入锁。
    def __init__(self) -> None:
        self._candidates: dict[str, IngestionCandidate] = {}
        self._lock = RLock()

    # 在锁保护下按候选标识读取记录，不存在时返回空值。
    def get(self, candidate_id: str) -> IngestionCandidate | None:
        with self._lock:
            return self._candidates.get(candidate_id)

    # 在锁保护下幂等保存候选，拒绝同一标识对应不同领域状态。
    def save(self, candidate: IngestionCandidate) -> IngestionCandidate:
        with self._lock:
            existing = self._candidates.get(candidate.candidate_id)
            if existing is not None and existing != candidate:
                raise ValueError(
                    "同一 candidate_id 不能覆盖为不同 domain state"
                )
            if existing is not None:
                return existing
            self._candidates[candidate.candidate_id] = candidate
            return candidate

    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        """仅供 Service 在合法状态迁移后替换冻结候选。"""
        with self._lock:
            if candidate.candidate_id not in self._candidates:
                raise KeyError(candidate.candidate_id)
            self._candidates[candidate.candidate_id] = candidate
            return candidate

    # 在锁保护下返回当前保存的候选数量。
    def count(self) -> int:
        with self._lock:
            return len(self._candidates)

    # 按状态和来源筛选候选，按申请时间及标识倒序返回分页结果。
    def list(
        self,
        *,
        status: IngestionCandidateStatus | None = None,
        source_type: KnowledgeSourceType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionCandidate]:
        if limit < 1 or offset < 0:
            raise ValueError("limit 必须为正整数且 offset 不能为负数")
        with self._lock:
            candidates = [
                candidate
                for candidate in self._candidates.values()
                if (status is None or candidate.status is status)
                and (
                    source_type is None
                    or candidate.document.source_type is source_type
                )
            ]
            candidates.sort(
                key=lambda item: (item.requested_at, item.candidate_id),
                reverse=True,
            )
            return candidates[offset : offset + limit]


__all__ = ["CandidateRepository", "InMemoryCandidateRepository"]
