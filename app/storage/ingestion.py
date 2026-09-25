"""Knowledge ingestion candidate 与 index operation 的 SQLAlchemy 适配。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import Index, Integer, Text, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.ingestion import IngestionCandidate, IngestionCandidateStatus, KnowledgeSourceType
from app.storage.database import get_engine, get_session_factory


class Base(DeclarativeBase): ...


# 映射入库候选表，保存查询字段和完整的候选契约 JSON。
class _IngestionCandidateRecord(Base):
    __tablename__ = "ingestion_candidates"

    candidate_id: Mapped[str] = mapped_column(Text, primary_key=True, nullable=True)
    status: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(Text)
    source_uri: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    requested_at: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(Text)
    candidate_json: Mapped[str] = mapped_column(Text)
    __table_args__ = (
        Index("idx_ingestion_candidates_status_requested", status, requested_at.desc()),
        Index("idx_ingestion_candidates_source_requested", source_type, requested_at.desc()),
    )


# 映射索引运行表，保存执行状态、时间、变更计数和错误信息。
class _IngestionIndexRunRecord(Base):
    __tablename__ = "ingestion_index_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=True)
    candidate_id: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    started_at: Mapped[str] = mapped_column(Text)
    completed_at: Mapped[str | None] = mapped_column(Text)
    added: Mapped[int | None] = mapped_column(Integer)
    updated: Mapped[int | None] = mapped_column(Integer)
    deleted: Mapped[int | None] = mapped_column(Integer)
    skipped: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    operation: Mapped[str] = mapped_column(Text, server_default="index")
    # 兼容旧库的 NULL；新 operation 必须由 repository 显式提供来源身份。
    source_id: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        Index("idx_ingestion_index_runs_candidate_id", candidate_id, id.desc()),
        Index("idx_ingestion_index_runs_status_id", status, id.desc()),
        {"sqlite_autoincrement": True},
    )


def init_ingestion_storage(db_path: Path | None = None) -> None:
    """保留已有数据建表，幂等补充 operation 并回填运行记录的来源身份。"""
    engine = get_engine(db_path)
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            # 串行检查与 ADD COLUMN，避免并发初始化重复添加字段。
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        Base.metadata.create_all(connection)
        if engine.dialect.name == "sqlite":
            columns = {row[1] for row in connection.exec_driver_sql(
                "PRAGMA table_info(ingestion_index_runs)"
            )}
            if "operation" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE ingestion_index_runs "
                    "ADD COLUMN operation TEXT NOT NULL DEFAULT 'index'"
                )
            if "source_id" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE ingestion_index_runs ADD COLUMN source_id TEXT"
                )
        with Session(bind=connection) as session:
            for run in session.scalars(select(_IngestionIndexRunRecord).where(
                _IngestionIndexRunRecord.source_id.is_(None)
            )):
                record = session.get(_IngestionCandidateRecord, run.candidate_id)
                if record is None:
                    raise ValueError("历史 corpus operation 缺失 candidate，无法回填 source_id")
                try:
                    candidate = IngestionCandidate.model_validate_json(record.candidate_json)
                except ValueError:
                    raise ValueError("历史 candidate 未通过 domain contract 校验，无法回填 source_id") from None
                if (record.candidate_id, record.status, record.source_type) != (
                    candidate.candidate_id, candidate.status.value,
                    candidate.document.source_type.value,
                ):
                    raise ValueError("历史 candidate 查询字段与 JSON 不一致")
                run.source_id = candidate.document.source_id
            session.flush()





class CandidatePersistenceError(RuntimeError):
    """持久 candidate 无法通过完整 domain contract 校验。"""


# 定义索引任务运行中、成功和失败三种持久化状态。
class IndexRunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class IndexRunOperation(str, Enum):
    """独立于 candidate 审批状态的知识库成员操作。"""

    INDEX = "index"
    REMOVE = "remove"


class IngestionIndexRun(BaseModel):
    """不含正文、payload 或 traceback 的持久知识库操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int
    candidate_id: str
    operation: IndexRunOperation = IndexRunOperation.INDEX
    status: IndexRunStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    added: int | None = Field(default=None, ge=0)
    updated: int | None = Field(default=None, ge=0)
    deleted: int | None = Field(default=None, ge=0)
    skipped: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=500)


# 生成包含 UTC 时区和毫秒精度的 ISO 时间字符串。
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# 校验分页数量为正数且偏移量非负。
def _page(limit: int, offset: int) -> None:
    if limit < 1 or offset < 0:
        raise ValueError("limit 必须为正整数且 offset 不能为负数")


class SQLiteCandidateRepository:
    """复用 application SQLite 的 CandidateRepository 持久实现。"""

    # 初始化数据库表结构，并取得候选仓库使用的共享会话工厂。
    def __init__(self, db_path: Path | None = None) -> None:
        init_ingestion_storage(db_path)
        self._sessions = get_session_factory(db_path)

    # 将候选转换为数据库字段，同时保存完整契约 JSON 和更新时间。
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

    # 从记录还原候选契约，并校验查询字段与 JSON 内容一致。
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

    # 按候选标识读取并校验持久化记录，不存在时返回空值。
    def get(self, candidate_id: str) -> IngestionCandidate | None:
        with self._sessions() as session:
            record = session.get(_IngestionCandidateRecord, candidate_id)
            return None if record is None else self._load(record)

    def delete(self, candidate_id: str) -> bool:
        """只删除候选及其审批 JSON，保留 corpus operation 历史。"""
        with self._sessions.begin() as session:
            record = session.get(_IngestionCandidateRecord, candidate_id)
            if record is None:
                return False
            session.delete(record)
            return True

    # 在事务中幂等插入候选，并拒绝已有标识对应不同领域状态。
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

    # 在事务中替换已有候选的持久化字段，候选不存在时抛出异常。
    def replace(self, candidate: IngestionCandidate) -> IngestionCandidate:
        with self._sessions.begin() as session:
            record = session.get(_IngestionCandidateRecord, candidate.candidate_id)
            if record is None:
                raise KeyError(candidate.candidate_id)
            for name, value in self._values(candidate).items():
                setattr(record, name, value)
        return candidate

    # 查询数据库中保存的候选总数。
    def count(self) -> int:
        with self._sessions() as session:
            statement = select(func.count()).select_from(_IngestionCandidateRecord)
            return int(session.scalar(statement) or 0)

    def load_latest_successfully_indexed(self) -> list[IngestionCandidate]:
        """按每个 source 最新成功操作恢复 INDEX；REMOVE 阻止回退到旧版本。"""
        statement = (
            select(_IngestionIndexRunRecord)
            .where(_IngestionIndexRunRecord.status == IndexRunStatus.SUCCESS.value)
            .order_by(_IngestionIndexRunRecord.id.desc())
        )
        seen: set[str] = set()
        candidates: list[IngestionCandidate] = []
        with self._sessions() as session:
            for run in session.scalars(statement):
                source_id = run.source_id
                if not source_id or not source_id.strip():
                    raise CandidatePersistenceError("持久 corpus operation 缺失 source_id")
                try:
                    operation = IndexRunOperation(run.operation)
                except ValueError:
                    raise CandidatePersistenceError("持久 corpus operation 无效") from None
                if source_id in seen:
                    continue
                seen.add(source_id)
                if operation is IndexRunOperation.INDEX:
                    record = session.get(_IngestionCandidateRecord, run.candidate_id)
                    if record is None:
                        raise CandidatePersistenceError("成功 INDEX 对应的 candidate 缺失")
                    candidate = self._load(record)
                    if candidate.document.source_id != source_id:
                        raise CandidatePersistenceError("INDEX source_id 与 candidate 不一致")
                    candidates.append(candidate)
        return candidates

    # 按状态和来源筛选候选，按申请时间及标识倒序返回校验后的分页记录。
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
    """保存同步 INDEX / REMOVE 状态与安全计数，不复制正文。"""

    # 初始化数据库表结构，并取得索引运行仓库使用的共享会话工厂。
    def __init__(self, db_path: Path | None = None) -> None:
        init_ingestion_storage(db_path)
        self._sessions = get_session_factory(db_path)

    # 将索引运行 ORM 记录校验并转换为领域返回模型。
    @staticmethod
    def _load(record: _IngestionIndexRunRecord) -> IngestionIndexRun:
        return IngestionIndexRun.model_validate(record, from_attributes=True)

    # 为候选创建运行中的索引记录，并返回数据库生成的运行标识。
    def create_running(
        self, candidate_id: str, *,
        source_id: str,
        operation: IndexRunOperation = IndexRunOperation.INDEX,
    ) -> IngestionIndexRun:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("新 corpus operation 必须提供非空 source_id")
        record = _IngestionIndexRunRecord(
            candidate_id=candidate_id,
            source_id=source_id,
            operation=operation.value,
            status=IndexRunStatus.RUNNING.value,
            started_at=_now(),
        )
        with self._sessions.begin() as session:
            session.add(record)
            session.flush()
            return self._load(record)

    # 将运行中的索引任务标记为成功，并保存变更计数。
    def succeed(self, run_id: int, **counts: int) -> IngestionIndexRun:
        return self._complete(run_id, IndexRunStatus.SUCCESS, counts=counts)

    # 将索引任务标记为失败，仅保存限制长度的错误码作为错误信息。
    def fail(self, run_id: int, *, error_code: str) -> IngestionIndexRun:
        safe_code = error_code[:128]
        return self._complete(
            run_id, IndexRunStatus.FAILED, error=(safe_code, safe_code)
        )

    # 在事务中结束运行中的索引任务，记录完成时间、计数和错误信息。
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

    # 返回指定候选最近一次索引运行记录，没有记录时返回空值。
    def latest(self, candidate_id: str) -> IngestionIndexRun | None:
        runs = self.list(candidate_id=candidate_id, limit=1)
        return runs[0] if runs else None

    def latest_successful_for_source(self, source_id: str) -> IngestionIndexRun | None:
        """按 logical source 返回最新成功操作，忽略失败及运行中操作。"""
        statement = (
            select(_IngestionIndexRunRecord)
            .where(
                _IngestionIndexRunRecord.source_id == source_id,
                _IngestionIndexRunRecord.status == IndexRunStatus.SUCCESS.value,
            )
            .order_by(_IngestionIndexRunRecord.id.desc())
            .limit(1)
        )
        with self._sessions() as session:
            record = session.scalar(statement)
            return None if record is None else self._load(record)

    # 按候选和状态筛选索引运行记录，并按运行标识倒序分页返回。
    def list(
        self,
        *,
        candidate_id: str | None = None,
        status: IndexRunStatus | None = None,
        operation: IndexRunOperation | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionIndexRun]:
        _page(limit, offset)
        filters = {}
        if candidate_id is not None:
            filters["candidate_id"] = candidate_id
        if status is not None:
            filters["status"] = status.value
        if operation is not None:
            filters["operation"] = operation.value
        statement = (
            select(_IngestionIndexRunRecord)
            .filter_by(**filters)
            .order_by(_IngestionIndexRunRecord.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with self._sessions() as session:
            return [self._load(record) for record in session.scalars(statement)]


__all__ = ["CandidatePersistenceError", "IndexRunOperation", "IndexRunStatus", "IngestionIndexRun",
           "SQLiteCandidateRepository", "SQLiteIndexRunRepository"]
