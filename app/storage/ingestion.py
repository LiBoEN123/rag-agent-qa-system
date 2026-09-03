"""Knowledge ingestion candidate 与 index operation 的 SQLite 适配。"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from app.ingestion import IngestionCandidate, IngestionCandidateStatus, KnowledgeSourceType
from app.storage.database import get_connection, init_db


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


@contextmanager
def _connection(path: Path | None) -> Iterator[sqlite3.Connection]:
    with closing(get_connection(path)) as connection, connection:
        yield connection


def _where(**filters: str | None) -> tuple[str, list[str]]:
    selected = [(name, value) for name, value in filters.items() if value is not None]
    clause = " WHERE " + " AND ".join(f"{name} = ?" for name, _ in selected)
    return (clause if selected else "", [value for _, value in selected])


class SQLiteCandidateRepository:
    """复用 application SQLite 的 CandidateRepository 持久实现。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        init_db(db_path)

    @staticmethod
    def _values(candidate: IngestionCandidate) -> tuple[str, ...]:
        document = candidate.document
        return (
            candidate.candidate_id,
            candidate.status.value,
            document.source_type.value,
            document.source_uri,
            document.content_hash,
            candidate.requested_at.isoformat(),
            _now(),
            candidate.model_dump_json(),
        )

    @staticmethod
    def _load(row: sqlite3.Row) -> IngestionCandidate:
        try:
            candidate = IngestionCandidate.model_validate_json(
                row["candidate_json"]
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise CandidatePersistenceError(
                "持久 candidate 未通过 domain contract 校验"
            ) from exc
        document = candidate.document
        if (
            row["candidate_id"], row["status"], row["source_type"]
        ) != (
            candidate.candidate_id, candidate.status.value,
            document.source_type.value,
        ):
            raise CandidatePersistenceError("candidate 查询字段与 JSON 不一致")
        return candidate

    def get(self, candidate_id: str) -> IngestionCandidate | None:
        with _connection(self._db_path) as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return None if row is None else self._load(row)

    def save(self, candidate: IngestionCandidate) -> IngestionCandidate:
        with _connection(self._db_path) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO ingestion_candidates(
                    candidate_id, status, source_type, source_uri, content_hash,
                    requested_at, updated_at, candidate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                self._values(candidate),
            )
        stored = self.get(candidate.candidate_id)
        if stored != candidate:
            raise ValueError(
                "同一 candidate_id 不能覆盖为不同 domain state"
            )
        return stored

    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        values = self._values(candidate)
        with _connection(self._db_path) as connection:
            cursor = connection.execute(
                """UPDATE ingestion_candidates SET status = ?, source_type = ?,
                    source_uri = ?, content_hash = ?, requested_at = ?,
                    updated_at = ?, candidate_json = ? WHERE candidate_id = ?""",
                (*values[1:], values[0]),
            )
        if cursor.rowcount != 1:
            raise KeyError(candidate.candidate_id)
        return candidate

    def count(self) -> int:
        with _connection(self._db_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM ingestion_candidates"
            ).fetchone()
        return int(row[0])

    def list(
        self,
        *,
        status: IngestionCandidateStatus | None = None,
        source_type: KnowledgeSourceType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionCandidate]:
        _page(limit, offset)
        where, parameters = _where(
            status=status.value if status else None,
            source_type=source_type.value if source_type else None,
        )
        with _connection(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_candidates" + where
                + " ORDER BY requested_at DESC, candidate_id DESC"
                " LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._load(row) for row in rows]


class SQLiteIndexRunRepository:
    """保存同步 index operation 状态与安全计数，不复制正文。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        init_db(db_path)

    @staticmethod
    def _load(row: sqlite3.Row) -> IngestionIndexRun:
        return IngestionIndexRun.model_validate(dict(row))

    def _get(self, run_id: int) -> IngestionIndexRun | None:
        with _connection(self._db_path) as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_index_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else self._load(row)

    def create_running(self, candidate_id: str) -> IngestionIndexRun:
        with _connection(self._db_path) as connection:
            cursor = connection.execute(
                """INSERT INTO ingestion_index_runs(
                    candidate_id, status, started_at
                ) VALUES (?, ?, ?)""",
                (candidate_id, IndexRunStatus.RUNNING.value, _now()),
            )
        return self._get(cursor.lastrowid)

    def succeed(self, run_id: int, **counts: int) -> IngestionIndexRun:
        return self._complete(run_id, IndexRunStatus.SUCCESS, counts=counts)

    def fail(
        self,
        run_id: int,
        *,
        error_code: str,
    ) -> IngestionIndexRun:
        safe_code = error_code[:128]
        return self._complete(
            run_id,
            IndexRunStatus.FAILED,
            error=(safe_code, safe_code),
        )

    def _complete(
        self,
        run_id: int,
        status: IndexRunStatus,
        *,
        counts: dict[str, int] | None = None,
        error: tuple[str, str] | None = None,
    ) -> IngestionIndexRun:
        values = counts or {}
        count_values = tuple(
            values.get(name)
            for name in ("added", "updated", "deleted", "skipped")
        )
        with _connection(self._db_path) as connection:
            cursor = connection.execute(
                """UPDATE ingestion_index_runs SET status = ?, completed_at = ?,
                    added = ?, updated = ?, deleted = ?, skipped = ?,
                    error_code = ?, error_message = ?
                WHERE id = ? AND status = ?""",
                (
                    status.value, _now(), *count_values,
                    *(error or (None, None)), run_id,
                    IndexRunStatus.RUNNING.value,
                ),
            )
        run = self._get(run_id)
        if cursor.rowcount != 1 or run is None:
            raise KeyError(run_id)
        return run

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
        where, parameters = _where(
            candidate_id=candidate_id,
            status=status.value if status else None,
        )
        with _connection(self._db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_index_runs" + where
                + " ORDER BY id DESC LIMIT ? OFFSET ?",
                (*parameters, limit, offset),
            ).fetchall()
        return [self._load(row) for row in rows]


__all__ = ["CandidatePersistenceError", "IndexRunStatus", "IngestionIndexRun",
           "SQLiteCandidateRepository", "SQLiteIndexRunRepository"]
