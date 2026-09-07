"""Knowledge ingestion candidate 与 index operation 的 SQLAlchemy 适配。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.ingestion import IngestionCandidate, IngestionCandidateStatus, KnowledgeSourceType
from app.storage.database import (
    _IngestionCandidateRecord,
    _IngestionIndexRunRecord,
    get_session_factory,
    init_db,
)


class CandidatePersistenceError(RuntimeError):
    """持久 candidate 无法通过完整 domain contract 校验。"""


class IndexRunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class IngestionIndexRun(BaseModel):
    """不含正文、payload 或 traceback 的持久 index operation。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int
    candidate_id: str
    status: IndexRunStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    added: int | None = Field(default=None, ge=0)
    updated: int | None = Field(default=None, ge=0)
    deleted: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=500)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _page(limit: int, offset: int) -> None:
    if limit < 1 or offset < 0:
        raise ValueError("limit 必须为正整数且 offset 不能为负数")


class SQLiteCandidateRepository:
    """复用 application SQLite 的 CandidateRepository 持久实现。"""

    def __init__(self, db_path: Path | None = None) -> None:
        init_db(db_path)
        self._sessions = get_session_factory(db_path)

    @staticmethod
    def _values(candidate: IngestionCandidate) -> dict[str, str]:
        document = candidate.document
        return {
            "candidate_id": candidate.candidate_id,
            "status": candidate.status.value,
            "source_type": document.source_type.value,
            "source_uri": document.source_uri,
            "content_hash": document.content_hash,
            "requested_at": candidate.requested_at.isoformat(),
            "updated_at": _now(),
            "candidate_json": candidate.model_dump_json(),
        }

    @staticmethod
    def _load(record: _IngestionCandidateRecord) -> IngestionCandidate:
        try:
            candidate = IngestionCandidate.model_validate_json(record.candidate_json)
        except (ValidationError, ValueError, TypeError) as exc:
            raise CandidatePersistenceError(
                "持久 candidate 未通过 domain contract 校验"
            ) from exc
        if (record.candidate_id, record.status, record.source_type) != (
            candidate.candidate_id,
            candidate.status.value,
            candidate.document.source_type.value,
        ):
            raise CandidatePersistenceError("candidate 查询字段与 JSON 不一致")
        return candidate

    def get(self, candidate_id: str) -> IngestionCandidate | None:
        with self._sessions() as session:
            record = session.get(_IngestionCandidateRecord, candidate_id)
            return None if record is None else self._load(record)

    def save(self, candidate: IngestionCandidate) -> IngestionCandidate:
        with self._sessions.begin() as session:
            session.execute(
                sqlite_insert(_IngestionCandidateRecord)
                .values(**self._values(candidate))
                .on_conflict_do_nothing(index_elements=["candidate_id"])
            )
            record = session.get(_IngestionCandidateRecord, candidate.candidate_id)
            if record is None:  # pragma: no cover - SQLite insert/get invariant
                raise CandidatePersistenceError("candidate 保存后无法读取")
            stored = self._load(record)
            if stored != candidate:
                raise ValueError("同一 candidate_id 不能覆盖为不同 domain state")
            return stored

    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        with self._sessions.begin() as session:
            record = session.get(_IngestionCandidateRecord, candidate.candidate_id)
            if record is None:
                raise KeyError(candidate.candidate_id)
            for name, value in self._values(candidate).items():
                setattr(record, name, value)
        return candidate

    def count(self) -> int:
        with self._sessions() as session:
            statement = select(func.count()).select_from(_IngestionCandidateRecord)
            return int(session.scalar(statement) or 0)

    def list(
        self,
        *,
        status: IngestionCandidateStatus | None = None,
        source_type: KnowledgeSourceType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionCandidate]:
        _page(limit, offset)
        filters = {}
        if status is not None:
            filters["status"] = status.value
        if source_type is not None:
            filters["source_type"] = source_type.value
        statement = (
            select(_IngestionCandidateRecord)
            .filter_by(**filters)
            .order_by(
                _IngestionCandidateRecord.requested_at.desc(),
                _IngestionCandidateRecord.candidate_id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        with self._sessions() as session:
            return [self._load(record) for record in session.scalars(statement)]


class SQLiteIndexRunRepository:
    """保存同步 index operation 状态与安全计数，不复制正文。"""

    def __init__(self, db_path: Path | None = None) -> None:
        init_db(db_path)
        self._sessions = get_session_factory(db_path)

    @staticmethod
    def _load(record: _IngestionIndexRunRecord) -> IngestionIndexRun:
        return IngestionIndexRun.model_validate(record, from_attributes=True)

    def create_running(self, candidate_id: str) -> IngestionIndexRun:
        record = _IngestionIndexRunRecord(
            candidate_id=candidate_id,
            status=IndexRunStatus.RUNNING.value,
            started_at=_now(),
        )
        with self._sessions.begin() as session:
            session.add(record)
            session.flush()
            return self._load(record)

    def succeed(self, run_id: int, **counts: int) -> IngestionIndexRun:
        return self._complete(run_id, IndexRunStatus.SUCCESS, counts=counts)

    def fail(self, run_id: int, *, error_code: str) -> IngestionIndexRun:
        safe_code = error_code[:128]
        return self._complete(
            run_id, IndexRunStatus.FAILED, error=(safe_code, safe_code)
        )

    def _complete(
        self,
        run_id: int,
        status: IndexRunStatus,
        *,
        counts: dict[str, int] | None = None,
        error: tuple[str, str] | None = None,
    ) -> IngestionIndexRun:
        with self._sessions.begin() as session:
            record = session.get(_IngestionIndexRunRecord, run_id)
            if record is None or record.status != IndexRunStatus.RUNNING.value:
                raise KeyError(run_id)
            record.status = status.value
            record.completed_at = _now()
            for name in ("added", "updated", "deleted", "skipped"):
                setattr(record, name, (counts or {}).get(name))
            record.error_code, record.error_message = error or (None, None)
            return self._load(record)

    def latest(self, candidate_id: str) -> IngestionIndexRun | None:
        runs = self.list(candidate_id=candidate_id, limit=1)
        return runs[0] if runs else None

    def list(
        self,
        *,
        candidate_id: str | None = None,
        status: IndexRunStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionIndexRun]:
        _page(limit, offset)
        filters = {}
        if candidate_id is not None:
            filters["candidate_id"] = candidate_id
        if status is not None:
            filters["status"] = status.value
        statement = (
            select(_IngestionIndexRunRecord)
            .filter_by(**filters)
            .order_by(_IngestionIndexRunRecord.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with self._sessions() as session:
            return [self._load(record) for record in session.scalars(statement)]


__all__ = ["CandidatePersistenceError", "IndexRunStatus", "IngestionIndexRun",
           "SQLiteCandidateRepository", "SQLiteIndexRunRepository"]
