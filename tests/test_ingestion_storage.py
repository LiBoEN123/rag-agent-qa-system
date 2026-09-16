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
from app.storage.database import _IngestionIndexRunRecord, get_session_factory, init_db
from app.storage.ingestion import (
    CandidatePersistenceError,
    IndexRunOperation,
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

    with sqlite3.connect(tmp_path / "app.sqlite3") as connection:
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
    success = repository.create_running("cand_success", source_id="src_success")
    completed = repository.succeed(
        success.id,
        added=1,
        updated=2,
        deleted=3,
        skipped=4,
    )
    failure = repository.create_running("cand_failure", source_id="src_failure")
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

    assert "source_id" not in completed.model_dump()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(ingestion_index_runs)")
        }
    assert "content" not in columns
    assert "candidate_json" not in columns


def test_latest_successfully_indexed_uses_success_history_without_page_limit(tmp_path):
    db_path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(db_path)
    domain = IngestionService(repository)
    runs = SQLiteIndexRunRepository(db_path)

    def approve(url, content):
        candidate = _propose(repository, url, content)
        return domain.approve(candidate.candidate_id, decided_by=REQUESTER)

    other = approve("https://example.com/other", "other source")
    v1 = approve("https://example.com/a", "v1")
    v2 = approve("https://example.com/a", "v2")
    failed_v3 = approve("https://example.com/a", "failed v3")
    approved_only = approve("https://example.com/unindexed", "approved only")
    failed_only = approve("https://example.com/failed", "failed only")
    for candidate in (other, v1, v2):
        runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
    for _ in range(51):
        runs.succeed(runs.create_running(v2.candidate_id, source_id=v2.document.source_id).id, skipped=1)
    for candidate in (failed_v3, failed_only):
        runs.fail(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, error_code="FAILED")
    runs.create_running(approved_only.candidate_id, source_id=approved_only.document.source_id)

    restarted = SQLiteCandidateRepository(db_path)
    assert restarted.load_latest_successfully_indexed() == [v2, other]
    # 最近一次成功重新索引旧 candidate，也应按 run 历史恢复该版本。
    runs.succeed(runs.create_running(v1.candidate_id, source_id=v1.document.source_id).id, added=1)
    assert restarted.load_latest_successfully_indexed() == [v1, other]


@pytest.mark.parametrize("remove_status", ["success", "failed", "running"])
def test_authoritative_source_uses_latest_successful_operation(tmp_path, remove_status):
    db_path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(db_path)
    domain = IngestionService(repository)
    runs = SQLiteIndexRunRepository(db_path)
    versions = []
    for content in ("v1", "v2"):
        candidate = _propose(repository, content=content)
        candidate = domain.approve(candidate.candidate_id, decided_by=REQUESTER)
        runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
        versions.append(candidate)
    # 旧 candidate 发起 REMOVE 也覆盖同 source 的最新版本。
    removal = runs.create_running(versions[0].candidate_id, source_id=versions[0].document.source_id, operation=IndexRunOperation.REMOVE)
    if remove_status == "success":
        runs.succeed(removal.id, deleted=1)
    elif remove_status == "failed":
        runs.fail(removal.id, error_code="REMOVE_OPERATION_FAILED")
    # 更新的失败或运行中 INDEX 都不能覆盖成功 REMOVE。
    runs.fail(runs.create_running(versions[1].candidate_id, source_id=versions[1].document.source_id).id, error_code="INDEX_OPERATION_FAILED")
    runs.create_running(versions[1].candidate_id, source_id=versions[1].document.source_id)
    restarted = SQLiteCandidateRepository(db_path)
    assert restarted.load_latest_successfully_indexed() == (
        [] if remove_status == "success" else [versions[1]]
    )
    assert [repository.get(item.candidate_id) for item in versions] == versions
    assert SQLiteIndexRunRepository(db_path).list(operation=IndexRunOperation.REMOVE)[0].id == removal.id
    runs.succeed(runs.create_running(versions[1].candidate_id, source_id=versions[1].document.source_id).id, added=1)
    assert restarted.load_latest_successfully_indexed() == [versions[1]]


def test_unknown_successful_operation_fails_closed(tmp_path):
    db_path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(db_path)
    candidate = _propose(repository)
    runs = SQLiteIndexRunRepository(db_path)
    runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE ingestion_index_runs SET operation = 'unknown'")
    with pytest.raises(CandidatePersistenceError, match="operation"):
        repository.load_latest_successfully_indexed()


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "mismatched_fields"])
def test_successful_candidate_restore_fails_closed_on_damaged_persistence(tmp_path, damage):
    db_path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(db_path)
    candidate = _propose(repository)
    runs = SQLiteIndexRunRepository(db_path)
    runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
    with sqlite3.connect(db_path) as connection:
        if damage == "missing":
            connection.execute("DELETE FROM ingestion_candidates")
        elif damage == "invalid_json":
            connection.execute("UPDATE ingestion_candidates SET candidate_json = '{}'")
        else:
            connection.execute("UPDATE ingestion_candidates SET source_type = 'github'")

    with pytest.raises(CandidatePersistenceError):
        repository.load_latest_successfully_indexed()


@pytest.mark.parametrize("repository_type", [InMemoryCandidateRepository, SQLiteCandidateRepository])
def test_delete_removes_only_candidate_and_allows_same_proposal(tmp_path, repository_type):
    repository = (repository_type(tmp_path / "app.sqlite3")
                  if repository_type is SQLiteCandidateRepository else repository_type())
    candidate = _propose(repository)
    other = _propose(repository, url="https://example.com/other")
    assert repository.delete(candidate.candidate_id) is True
    assert repository.delete(candidate.candidate_id) is False
    assert repository.get(candidate.candidate_id) is None
    assert repository.list() == [other]
    assert repository.count() == 1
    assert _propose(repository) == candidate


@pytest.mark.parametrize("source_id", [None, "", "  ", 123])
def test_new_run_requires_nonempty_source_id(tmp_path, source_id):
    runs = SQLiteIndexRunRepository(tmp_path / "app.sqlite3")
    with pytest.raises(ValueError, match="source_id"):
        runs.create_running("candidate", source_id=source_id)
    assert runs.list() == []


def test_latest_successful_query_is_scoped_to_source_and_ignores_incomplete_runs(tmp_path):
    runs = SQLiteIndexRunRepository(tmp_path / "app.sqlite3")
    indexed = runs.succeed(runs.create_running("v1", source_id="source_a").id)
    runs.succeed(runs.create_running("other", source_id="source_b", operation=IndexRunOperation.REMOVE).id)
    runs.fail(runs.create_running("v2", source_id="source_a", operation=IndexRunOperation.REMOVE).id, error_code="FAILED")
    runs.create_running("v3", source_id="source_a", operation=IndexRunOperation.REMOVE)
    assert runs.latest_successful_for_source("source_a") == indexed
    assert runs.latest_successful_for_source("source_b").operation is IndexRunOperation.REMOVE
    assert runs.latest_successful_for_source("source_missing") is None


def test_source_identity_survives_deleted_versions_and_repository_restart(tmp_path):
    path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(path)
    runs = SQLiteIndexRunRepository(path)
    versions = [_propose(repository, content=content) for content in ("v1", "v2", "v3")]
    source_id = versions[0].document.source_id
    for candidate in versions[:2]:
        runs.succeed(runs.create_running(candidate.candidate_id, source_id=source_id).id)
    remove = runs.succeed(runs.create_running(
        versions[0].candidate_id, source_id=source_id, operation=IndexRunOperation.REMOVE,
    ).id)
    history = runs.list()
    with get_session_factory(path)() as session:
        for run in history:
            assert session.get(_IngestionIndexRunRecord, run.id).source_id == source_id
    for candidate in versions:
        assert runs.latest_successful_for_source(source_id) == remove
        assert repository.delete(candidate.candidate_id)
    assert SQLiteCandidateRepository(path).load_latest_successfully_indexed() == []
    restarted_runs = SQLiteIndexRunRepository(path)
    assert restarted_runs.list() == history
    assert restarted_runs.latest_successful_for_source(source_id) == remove


@pytest.mark.parametrize("operation", list(IndexRunOperation))
def test_legacy_source_id_backfill_is_idempotent(tmp_path, operation):
    path = tmp_path / "legacy.sqlite3"
    repository = SQLiteCandidateRepository(path)
    candidate = _propose(repository)
    runs = SQLiteIndexRunRepository(path)
    for status in IndexRunStatus:
        run = runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id, operation=operation)
        if status is IndexRunStatus.SUCCESS:
            runs.succeed(run.id)
        elif status is IndexRunStatus.FAILED:
            runs.fail(run.id, error_code="FAILED")
    history = runs.list()
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE ingestion_index_runs DROP COLUMN source_id")
    for _ in range(3):
        init_db(path)
    with get_session_factory(path)() as session:
        for run in history:
            assert session.get(_IngestionIndexRunRecord, run.id).source_id == candidate.document.source_id
    assert SQLiteIndexRunRepository(path).list() == history
    assert repository.get(candidate.candidate_id) == candidate


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "mismatched_fields"])
def test_legacy_backfill_fails_closed_and_does_not_invent_source_id(tmp_path, damage):
    path = tmp_path / "legacy.sqlite3"
    repository = SQLiteCandidateRepository(path)
    candidate = _propose(repository)
    runs = SQLiteIndexRunRepository(path)
    runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ingestion_index_runs SET source_id = NULL")
        if damage == "missing":
            connection.execute("DELETE FROM ingestion_candidates")
        elif damage == "invalid_json":
            connection.execute("UPDATE ingestion_candidates SET candidate_json = '{}'")
        else:
            connection.execute("UPDATE ingestion_candidates SET source_type = 'github'")
    with pytest.raises(ValueError):
        init_db(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT source_id FROM ingestion_index_runs").fetchone() == (None,)


@pytest.mark.parametrize("source_id", [None, "", "src_wrong"])
def test_authoritative_index_fails_closed_on_invalid_source_identity(tmp_path, source_id):
    path = tmp_path / "app.sqlite3"
    repository = SQLiteCandidateRepository(path)
    candidate = _propose(repository)
    runs = SQLiteIndexRunRepository(path)
    runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ingestion_index_runs SET source_id = ?", (source_id,))
    with pytest.raises(CandidatePersistenceError, match="source_id"):
        repository.load_latest_successfully_indexed()


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

    with sqlite3.connect(db_path) as upgraded:
        tables = {
            row[0]
            for row in upgraded.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert upgraded.execute("SELECT content FROM messages").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT question FROM agent_runs").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT action FROM agent_steps").fetchone()[0] == "kept"
        assert upgraded.execute("SELECT session_id FROM sessions").fetchone()[0] == "s1"
    assert {"ingestion_candidates", "ingestion_index_runs"} <= tables
