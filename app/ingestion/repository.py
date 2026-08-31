"""Phase 9 的最小候选 repository contract 与内存实现。"""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from app.ingestion.contracts import IngestionCandidate


class CandidateRepository(Protocol):
    """候选存储边界；Phase 9 不提供持久数据库实现。"""

    def get(self, candidate_id: str) -> IngestionCandidate | None:
        """按稳定 candidate ID 查询。"""

    def save(self, candidate: IngestionCandidate) -> IngestionCandidate:
        """保存或幂等返回完全相同的候选。"""

    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        """在 Service 已验证状态迁移后替换现有候选。"""

    def count(self) -> int:
        """返回逻辑候选数量。"""


class InMemoryCandidateRepository:
    """线程安全且不会为同一 candidate_id 建立逻辑重复的内存存储。"""

    def __init__(self) -> None:
        self._candidates: dict[str, IngestionCandidate] = {}
        self._lock = RLock()

    def get(self, candidate_id: str) -> IngestionCandidate | None:
        with self._lock:
            return self._candidates.get(candidate_id)

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

    def count(self) -> int:
        with self._lock:
            return len(self._candidates)


__all__ = ["CandidateRepository", "InMemoryCandidateRepository"]
