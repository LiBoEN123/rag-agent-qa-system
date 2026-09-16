"""SLIM-4: frozen legacy schema、共享连接生命周期和存储行为回归。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import event

from app.ingestion import (
    InMemoryCandidateRepository,
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    build_knowledge_document,
)
from app.storage.database import (
    Base,
    clear_storage_engine_cache,
    get_engine,
    get_session_factory,
    init_db,
    now_iso,
)
from app.storage.ingestion import (
    IndexRunOperation,
    IndexRunStatus,
    SQLiteCandidateRepository,
    SQLiteIndexRunRepository,
)
from app.storage.memory import (
    append_message_sqlite,
    clear_session_sqlite,
    get_history_sqlite,
)


LEGACY_SCHEMA = Path(__file__).parent / "fixtures" / "application_phase12.sql"
LOCAL_TIME = "2026-09-05T08:00:00.000"
UTC_TIME = datetime(2026, 9, 5, tzinfo=timezone.utc)
REQUESTER = IngestionRequester(
    principal_id="storage-test",
    principal_type=IngestionPrincipalType.SERVICE,
    source="pytest",
)


def _candidate(uri="https://example.com/legacy"):
    return IngestionService(InMemoryCandidateRepository()).propose(
        document=build_knowledge_document(
            source_type="firecrawl", source_uri=uri,
            title="旧文档", content="旧正文", fetched_at=UTC_TIME,
        ),
        requester=REQUESTER,
        requested_at=UTC_TIME,
    )


def _schema(path):
    """直接检查 SQLite，包括索引方向、隐式 PK 索引和外键。"""
    with closing(sqlite3.connect(path)) as connection:
        tables = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
        ).fetchall()
        result = {}
        for name, sql in tables:
            indexes = {}
            for index in connection.execute(f'PRAGMA index_list("{name}")'):
                indexes[index[1]] = {
                    "flags": index[2:],
                    "columns": connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall(),
                    "details": connection.execute(f'PRAGMA index_xinfo("{index[1]}")').fetchall(),
                }
            result[name] = {
                "columns": connection.execute(f'PRAGMA table_info("{name}")').fetchall(),
                "indexes": indexes,
                "foreign_keys": connection.execute(f'PRAGMA foreign_key_list("{name}")').fetchall(),
                "autoincrement": "AUTOINCREMENT" in sql.upper(),
            }
        return result


def _rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return {
            name: connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
            for name in _schema(path)
        }


@pytest.fixture
def legacy_db(tmp_path):
    """仅用旧 raw SQL 建库和写入，不能依赖新 metadata 或 storage 写函数。"""
    path = tmp_path / "legacy.sqlite3"
    candidate = _candidate()
    document = candidate.document
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(LEGACY_SCHEMA.read_text())
        connection.execute("INSERT INTO sessions VALUES (?, ?, ?)", ("s1", LOCAL_TIME, LOCAL_TIME))
        connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", [
            (7, "s1", "user", "旧问题", LOCAL_TIME),
            (8, "s1", "assistant", "旧回答", LOCAL_TIME),
        ])
        connection.execute("INSERT INTO agent_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            "t1", "s1", "旧问题", "改写", "旧回答", "OK", "完成", LOCAL_TIME, 12,
        ))
        connection.execute("INSERT INTO agent_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            11, "t1", 2, "tool_call", "search", '{"query": "旧问题"}',
            '{"step_id": 2}', "success", 5, LOCAL_TIME,
        ))
        connection.execute("INSERT INTO ingestion_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
            candidate.candidate_id, candidate.status.value, document.source_type.value,
            document.source_uri, document.content_hash, candidate.requested_at.isoformat(),
            UTC_TIME.isoformat(), candidate.model_dump_json(),
        ))
        connection.execute(
            "INSERT INTO ingestion_index_runs(id, candidate_id, status, started_at) VALUES (?, ?, ?, ?)",
            (17, candidate.candidate_id, "running", UTC_TIME.isoformat()),
        )
    return path, candidate


def test_fresh_schema_matches_frozen_raw_sql_and_shared_metadata(tmp_path, legacy_db):
    path = tmp_path / "fresh.sqlite3"
    init_db(path)
    retained_schema = {
        name: table for name, table in _schema(legacy_db[0]).items()
        if name not in {"agent_runs", "agent_steps"}
    }
    columns = retained_schema["ingestion_index_runs"]["columns"]
    columns.append((len(columns), "operation", "TEXT", 1, "'index'", 0))
    columns.append((len(columns), "source_id", "TEXT", 0, None, 0))
    assert _schema(path) == retained_schema
    assert set(_schema(path)) == set(Base.metadata.tables) == {
        "sessions", "messages",
        "ingestion_candidates", "ingestion_index_runs",
    }
    assert sum(len(table["indexes"]) for table in _schema(path).values()) == 7
    metadata_only = tmp_path / "metadata-only.sqlite3"
    Base.metadata.create_all(get_engine(metadata_only))
    assert _schema(metadata_only) == retained_schema


def test_legacy_reads_and_repeated_init_preserve_all_existing_rows_and_schema(legacy_db):
    path, candidate = legacy_db
    before, schema_before = _rows(path), _schema(path)
    before["ingestion_index_runs"] = [row + ("index", candidate.document.source_id) for row in before["ingestion_index_runs"]]
    columns = schema_before["ingestion_index_runs"]["columns"]
    columns.append((len(columns), "operation", "TEXT", 1, "'index'", 0))
    columns.append((len(columns), "source_id", "TEXT", 0, None, 0))
    for _ in range(3):
        init_db(path)
    assert get_history_sqlite("s1", db_path=path) == [
        {"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"},
    ]
    assert SQLiteCandidateRepository(path).get(candidate.candidate_id) == candidate
    run = SQLiteIndexRunRepository(path).latest(candidate.candidate_id)
    assert (run.id, run.status, run.started_at) == (17, IndexRunStatus.RUNNING, UTC_TIME)
    assert run.operation is IndexRunOperation.INDEX
    assert _rows(path) == before
    assert _schema(path) == schema_before


def test_legacy_writes_replace_delete_and_restart_without_migration(legacy_db):
    path, candidate = legacy_db
    rows_before, schema_before = _rows(path), _schema(path)
    columns = schema_before["ingestion_index_runs"]["columns"]
    columns.append((len(columns), "operation", "TEXT", 1, "'index'", 0))
    columns.append((len(columns), "source_id", "TEXT", 0, None, 0))
    init_db(path)
    append_message_sqlite("s1", "user", "追加", path)
    append_message_sqlite("s2", "user", "保留", path)
    candidates, runs = SQLiteCandidateRepository(path), SQLiteIndexRunRepository(path)
    approved = IngestionService(candidates).approve(
        candidate.candidate_id, decided_by=REQUESTER, decided_at=UTC_TIME,
    )
    second = candidates.save(_candidate("https://example.com/new"))
    completed = runs.succeed(17, added=1, updated=0, deleted=0, skipped=0)
    running = runs.create_running(second.candidate_id, source_id=second.document.source_id)
    failed = runs.fail(running.id, error_code="SAFE_ERROR")
    assert running.id > 17
    assert get_history_sqlite("s1", limit=1, db_path=path)[0]["content"] == "追加"
    clear_session_sqlite("s1", path)
    clear_storage_engine_cache()
    assert get_history_sqlite("s1", db_path=path) == []
    assert get_history_sqlite("s2", db_path=path) == [{"role": "user", "content": "保留"}]
    assert SQLiteCandidateRepository(path).get(candidate.candidate_id) == approved
    assert SQLiteCandidateRepository(path).get(second.candidate_id) == second
    assert SQLiteIndexRunRepository(path).list() == [failed, completed]
    assert _schema(path) == schema_before
    rows_after = _rows(path)
    for name in ("agent_runs", "agent_steps"):
        assert rows_after[name] == rows_before[name]


def test_legacy_success_defaults_to_index_and_remains_authoritative(legacy_db):
    path, candidate = legacy_db
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE ingestion_index_runs SET status = 'success', completed_at = ?",
            (UTC_TIME.isoformat(),),
        )
    repository = SQLiteCandidateRepository(path)
    assert repository.load_latest_successfully_indexed() == [candidate]
    runs = SQLiteIndexRunRepository(path)
    assert runs.latest(candidate.candidate_id).operation is IndexRunOperation.INDEX
    remove = runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id, operation=IndexRunOperation.REMOVE)
    runs.succeed(remove.id, deleted=1)
    for _ in range(3):
        init_db(path)
    assert SQLiteCandidateRepository(path).load_latest_successfully_indexed() == []
    assert runs.latest(candidate.candidate_id).operation is IndexRunOperation.REMOVE


@pytest.mark.parametrize("session_id, content", [(None, "x"), (" ", "x"), ("s", ""), ("s", " ")])
def test_memory_empty_inputs_are_noop_even_with_invalid_role(tmp_path, session_id, content):
    path = tmp_path / "absent.sqlite3"
    append_message_sqlite(session_id, "invalid", content, path)
    clear_session_sqlite(None, path)
    assert get_history_sqlite(None, db_path=path) == []
    assert get_history_sqlite("s", limit=0, db_path=path) == []
    assert get_history_sqlite("s", limit=-1, db_path=path) == []
    assert not path.exists()


def test_memory_upsert_timestamps_and_restart(tmp_path, monkeypatch):
    path = tmp_path / "app.sqlite3"
    later = "2026-09-05T09:00:00.000"
    timestamps = iter([LOCAL_TIME, later])
    monkeypatch.setattr("app.storage.memory.now_iso", lambda: next(timestamps))
    append_message_sqlite(" s ", "user", " first ", path)
    clear_storage_engine_cache()
    append_message_sqlite("s", "assistant", "second", path)
    assert _rows(path)["sessions"] == [("s", LOCAL_TIME, later)]
    assert get_history_sqlite("s", db_path=path) == [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "second"},
    ]
    clear_session_sqlite("s", path)
    assert _rows(path)["sessions"] == []
    assert _rows(path)["messages"] == []
    assert datetime.fromisoformat(now_iso()).tzinfo is None


def test_message_insert_failure_rolls_back_session_upsert(tmp_path):
    path = tmp_path / "app.sqlite3"
    append_message_sqlite("s", "user", "before", path)
    before = _rows(path)
    engine = get_engine(path)

    def reject_message(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO messages"):
            raise RuntimeError("injected insert failure")

    event.listen(engine, "before_cursor_execute", reject_message)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            append_message_sqlite("s", "assistant", "after", path)
    finally:
        event.remove(engine, "before_cursor_execute", reject_message)
    assert _rows(path) == before


def test_shared_engine_factory_paths_threads_and_dispose(tmp_path, monkeypatch, settings_env):
    path = tmp_path / "nested" / "中文?#.sqlite3"
    settings_env(sqlite_db_path=path)
    monkeypatch.chdir(tmp_path)
    relative = Path("nested") / path.name
    with ThreadPoolExecutor(max_workers=4) as pool:
        engines = list(pool.map(get_engine, [path, relative, None, path.parent / ".." / "nested" / path.name]))
    assert all(engine is engines[0] for engine in engines)
    candidates, runs = SQLiteCandidateRepository(relative), SQLiteIndexRunRepository(path)
    factory = get_session_factory()
    assert candidates._sessions is runs._sessions is factory is get_session_factory(relative)
    with factory() as session:
        assert session.get_bind() is engines[0]
        assert session.expire_on_commit is False
    append_message_sqlite("s", "user", "main thread", path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(append_message_sqlite, "s", "assistant", "worker thread", relative).result()
    other = tmp_path / "other.sqlite3"
    assert get_engine(other) is not engines[0]
    assert get_history_sqlite("s", db_path=other) == []
    disposed = []
    event.listen(engines[0], "engine_disposed", lambda engine: disposed.append(engine))
    clear_storage_engine_cache()
    assert disposed == [engines[0]]
    assert get_engine(path) is not engines[0]
    assert get_session_factory(path) is not factory
    assert len(get_history_sqlite("s", db_path=path)) == 2


@pytest.mark.parametrize("table, values", [
    ("messages", {"session_id": "s", "role": "user", "content": "q", "created_at": LOCAL_TIME}),
    ("ingestion_index_runs", {"candidate_id": "c", "status": "running", "started_at": UTC_TIME.isoformat()}),
])
@pytest.mark.parametrize("legacy", [False, True])
def test_autoincrement_never_reuses_deleted_maximum_id(tmp_path, legacy_db, table, values, legacy):
    path = legacy_db[0] if legacy else tmp_path / "fresh.sqlite3"
    init_db(path)
    table = Base.metadata.tables[table]
    with get_session_factory(path).begin() as session:
        previous = session.execute(table.insert().values(**values)).inserted_primary_key[0]
        session.execute(table.delete().where(table.c.id == previous))
        current = session.execute(table.insert().values(**values)).inserted_primary_key[0]
    assert current > previous
