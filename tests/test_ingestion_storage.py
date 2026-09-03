"""Phase 12 ingestion SQLite persistence 与旧库兼容测试。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from app.ingestion import (
    InMemoryCandidateRepository,
    IngestionCandidateStatus,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeSourceType,
)
from app.sources import FirecrawlPagePayload, FirecrawlPageSourceAdapter
from app.storage.database import get_connection, init_db
from app.storage.ingestion import (
    CandidatePersistenceError,
    IndexRunStatus,
    SQLiteCandidateRepository,
    SQLiteIndexRunRepository,
)


FIXED_TIME = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)
REQUESTER = IngestionRequester(
    principal_id="ingestion-storage-test",
    principal_type=IngestionPrincipalType.SERVICE,
    source="pytest",
)


def _document(url: str = "https://example.com/a", content: str = "body"):
    return FirecrawlPageSourceAdapter().to_document(
        FirecrawlPagePayload(
            requested_url=url,
            final_url=url,
            title=url.rsplit("/", 1)[-1],
            content=content,
            mime_type="text/html",
            status_code=200,
        ),
        fetched_at=FIXED_TIME,
    )


def _propose(repository, url="https://example.com/a", content="body"):
    return IngestionService(repository).propose(
        document=_document(url, content),
        requester=REQUESTER,
        requested_at=FIXED_TIME,
    )


def test_candidate_pending_approved_and_rejected_survive_repository_restart(tmp_path):
    db_path = tmp_path / "app.sqlite3"
    first_repository = SQLiteCandidateRepository(db_path)
    pending = _propose(first_repository)
    rejected_pending = _propose(first_repository, "https://example.com/reject")

    second_repository = SQLiteCandidateRepository(db_path)
    assert second_repository.get(pending.candidate_id) == pending
    assert second_repository.get(pending.candidate_id).status is IngestionCandidateStatus.PENDING

    approved = IngestionService(second_repository).approve(
        pending.candidate_id,
        decided_by=REQUESTER,
        decided_at=FIXED_TIME,
    )
    rejected = IngestionService(second_repository).reject(
        rejected_pending.candidate_id,
        decided_by=REQUESTER,
        reason="not suitable",
        decided_at=FIXED_TIME,
    )

    third_repository = SQLiteCandidateRepository(db_path)
    assert third_repository.count() == 2
    assert third_repository.get(approved.candidate_id) == approved
    assert third_repository.get(rejected.candidate_id) == rejected
    assert third_repository.get(approved.candidate_id).status is IngestionCandidateStatus.APPROVED
    assert third_repository.get(rejected.candidate_id).status is IngestionCandidateStatus.REJECTED


def test_candidate_list_filters_and_paginates_without_skipping_validation(tmp_path):
    repository = SQLiteCandidateRepository(tmp_path / "app.sqlite3")
    first = _propose(repository, "https://example.com/a")
    second = _propose(repository, "https://example.com/b")
    approved = IngestionService(repository).approve(
        first.candidate_id,
        decided_by=REQUESTER,
        decided_at=FIXED_TIME,
    )

    assert repository.list(status=IngestionCandidateStatus.APPROVED) == [approved]
    assert repository.list(source_type=KnowledgeSourceType.FIRECRAWL, limit=1) == [second]
    assert repository.list(source_type=KnowledgeSourceType.FIRECRAWL, limit=1, offset=1) == [approved]

    with get_connection(tmp_path / "app.sqlite3") as connection:
        connection.execute(
            "UPDATE ingestion_candidates SET candidate_json = ? WHERE candidate_id = ?",
            ("{}", approved.candidate_id),
        )
    with pytest.raises(CandidatePersistenceError):
        repository.get(approved.candidate_id)


def test_sqlalchemy_repository_reads_and_updates_raw_phase12_schema(tmp_path):
    db_path = tmp_path / "phase12.sqlite3"
    candidate = _propose(InMemoryCandidateRepository())
    document = candidate.document
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE ingestion_candidates (
                candidate_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                source_type TEXT NOT NULL, source_uri TEXT NOT NULL,
                content_hash TEXT NOT NULL, requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, candidate_json TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO ingestion_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candidate.candidate_id,
                candidate.status.value,
                document.source_type.value,
                document.source_uri,
                document.content_hash,
                candidate.requested_at.isoformat(),
                candidate.requested_at.isoformat(),
                candidate.model_dump_json(),
            ),
        )

    repository = SQLiteCandidateRepository(db_path)
    assert repository.get(candidate.candidate_id) == candidate
    assert repository.save(candidate) == candidate
    approved = IngestionService(repository).approve(
        candidate.candidate_id,
        decided_by=REQUESTER,
        decided_at=FIXED_TIME,
    )
    assert repository.list(status=IngestionCandidateStatus.APPROVED) == [approved]
    with pytest.raises(ValueError, match="不能覆盖"):
        repository.save(candidate)
    with pytest.raises(KeyError):
        repository.replace(_propose(InMemoryCandidateRepository(), url="https://x.test"))


def test_index_runs_persist_success_and_safe_failure_without_content(tmp_path):
    db_path = tmp_path / "app.sqlite3"
    repository = SQLiteIndexRunRepository(db_path)
    success = repository.create_running("cand_success")
    completed = repository.succeed(
        success.id,
        added=1,
        updated=2,
        deleted=3,
        skipped=4,
    )
    failure = repository.create_running("cand_failure")
    failed = repository.fail(
        failure.id,
        error_code="SOURCE_RUNTIME_RELOAD_FAILED",
    )

    restarted = SQLiteIndexRunRepository(db_path)
    assert restarted.latest("cand_success") == completed
    assert restarted.latest("cand_failure") == failed
    assert completed.status is IndexRunStatus.SUCCESS
    assert (completed.added, completed.updated, completed.deleted, completed.skipped) == (1, 2, 3, 4)
    assert failed.status is IndexRunStatus.FAILED
    assert restarted.list(status=IndexRunStatus.SUCCESS, limit=1) == [completed]
    assert restarted.list(limit=1, offset=1) == [completed]

    with get_connection(db_path) as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(ingestion_index_runs)")
        }
    assert "content" not in columns
    assert "candidate_json" not in columns


def test_init_db_upgrades_existing_database_without_changing_old_rows(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE agent_runs (
            trace_id TEXT PRIMARY KEY, session_id TEXT, question TEXT NOT NULL,
            rewritten_question TEXT, final_answer TEXT, status TEXT, reason TEXT,
            created_at TEXT NOT NULL, total_cost_ms INTEGER DEFAULT 0
        );
        CREATE TABLE agent_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL,
            step_id INTEGER, action TEXT, tool_name TEXT, tool_args_json TEXT,
            observation_json TEXT, status TEXT, cost_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        INSERT INTO sessions VALUES ('s1', 'before', 'before');
        INSERT INTO messages(session_id, role, content, created_at)
        VALUES ('s1', 'user', 'kept', 'before');
        INSERT INTO agent_runs(trace_id, question, created_at)
        VALUES ('t1', 'kept', 'before');
        INSERT INTO agent_steps(trace_id, action, created_at)
        VALUES ('t1', 'kept', 'before');
        """
    )
    connection.commit()
    connection.close()

    init_db(db_path)

    with get_connection(db_path) as upgraded:
        tables = {
            row["name"]
            for row in upgraded.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert upgraded.execute("SELECT content FROM messages").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT question FROM agent_runs").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT action FROM agent_steps").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT session_id FROM sessions").fetchone()[0] == "s1"
    assert {"ingestion_candidates", "ingestion_index_runs"} <= tables
