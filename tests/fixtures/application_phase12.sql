-- Frozen raw schema from SLIM-3 HEAD 5e6a5ce4846e2083cbb7bdb4321b60e82cda66b6.
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS agent_runs (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT,
    question TEXT NOT NULL,
    rewritten_question TEXT,
    final_answer TEXT,
    status TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    total_cost_ms INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    step_id INTEGER,
    action TEXT,
    tool_name TEXT,
    tool_args_json TEXT,
    observation_json TEXT,
    status TEXT,
    cost_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_steps_trace_id_id
ON agent_steps(trace_id, id);

CREATE TABLE IF NOT EXISTS ingestion_candidates (
    candidate_id TEXT PRIMARY KEY,
    status TEXT NOT NULL, source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL, content_hash TEXT NOT NULL,
    requested_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    candidate_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ingestion_candidates_status_requested
ON ingestion_candidates(status, requested_at DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_candidates_source_requested
ON ingestion_candidates(source_type, requested_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_index_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL, status TEXT NOT NULL,
    started_at TEXT NOT NULL, completed_at TEXT,
    added INTEGER, updated INTEGER,
    deleted INTEGER, skipped INTEGER,
    error_code TEXT, error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_index_runs_candidate_id
ON ingestion_index_runs(candidate_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_index_runs_status_id
ON ingestion_index_runs(status, id DESC);
