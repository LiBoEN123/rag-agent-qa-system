"""结构的 normalization、identity、持久化与旧 candidate 兼容。"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from app.ingestion import IngestionCandidate, IngestionRequester, IngestionService, KnowledgeDocument
from app.ingestion.document_blocks import ParsedDocument, TableBlock, TableCell, TextBlock
from app.ingestion.factory import build_knowledge_document
from app.ingestion.identity import build_content_hash, build_structure_hash
from app.ingestion.normalization import prepare_structured_content
from app.rag.indexing import candidate_to_langchain_documents
from app.storage.database import clear_storage_engine_cache
from app.storage.ingestion import CandidatePersistenceError, SQLiteCandidateRepository, SQLiteIndexRunRepository


REQUESTER = IngestionRequester(principal_id="structure-test", principal_type="user", source="pytest")


def _structure(*, merged=False, text="value"):
    return ParsedDocument(blocks=(TableBlock(
        block_id="b0", order=0, row_count=1, column_count=2,
        cells=(TableCell(text=text, row_index=0, column_index=0, column_span=2),) if merged else (
            TableCell(text=text, row_index=0, column_index=0),
            TableCell(text="", row_index=0, column_index=1),
        ),
    ),))


def _document(structure=None):
    return build_knowledge_document(
        source_type="upload", source_uri="upload://local/stable", title="table",
        content="legacy content", structured_content=structure, metadata={"file_type": "docx"},
    )


def test_same_text_distinct_structure_changes_only_structured_candidate_identity():
    service = IngestionService()
    first = _document(_structure())
    second = _document(_structure(merged=True))
    assert first.content == second.content
    assert first.content_hash == second.content_hash == build_content_hash(first.content)
    assert first.source_id == second.source_id
    assert first.document_uid == second.document_uid
    assert first.structure_hash != second.structure_hash
    a = service.propose(document=first, requester=REQUESTER)
    b = service.propose(document=second, requester=REQUESTER)
    assert a.candidate_id != b.candidate_id
    assert service.propose(document=_document(_structure()), requester=REQUESTER) == a


def test_structure_hash_is_canonical_across_key_cell_order_and_processes():
    document = _document(_structure())
    payload = document.structured_content.model_dump(mode="json")
    payload["blocks"][0]["cells"].reverse()
    payload = json.loads(json.dumps(payload, sort_keys=True))
    normalized, _ = prepare_structured_content(payload)
    assert build_structure_hash(normalized) == document.structure_hash
    code = (
        "import sys; from app.ingestion.normalization import prepare_structured_content; "
        "from app.ingestion.identity import build_structure_hash; import json; "
        "print(build_structure_hash(prepare_structured_content(json.load(sys.stdin))[0]))"
    )
    for seed in ("1", "42"):
        output = subprocess.check_output(
            [sys.executable, "-c", code], input=json.dumps(payload), text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert output.strip() == document.structure_hash


def test_all_structured_text_slots_are_normalized_and_redacted_before_persistence(tmp_path):
    structure = ParsedDocument(blocks=(
        TextBlock(block_id="b0", order=0, text="Cafe\u0301\r\nAuthorization: Bearer block-secret  "),
        TableBlock(block_id="b1", order=1, caption="api_key=caption-secret", row_count=1, column_count=1,
                   cells=(TableCell(text="password=cell-secret\r\ntrail  ", row_index=0, column_index=0),)),
    ))
    document = _document(structure)
    assert document.content_redacted
    assert document.structured_content.blocks[0].text.startswith("Café\n")
    assert document.structured_content.blocks[1].cells[0].text.endswith("\ntrail")
    candidate = IngestionService(SQLiteCandidateRepository(tmp_path / "app.db")).propose(
        document=document, requester=REQUESTER,
    )
    with sqlite3.connect(tmp_path / "app.db") as connection:
        persisted = connection.execute("SELECT candidate_json FROM ingestion_candidates").fetchone()[0]
    for secret in ("block-secret", "caption-secret", "cell-secret"):
        assert secret not in persisted
    assert IngestionCandidate.model_validate_json(persisted) == candidate
    # 直接 KnowledgeDocument validation 也必须执行同一边界。
    direct = document.model_dump()
    direct["structured_content"] = structure.model_dump()
    assert KnowledgeDocument.model_validate(direct) == document


def test_sqlite_reload_and_successful_rebuild_input_keep_complete_structure(tmp_path):
    path = tmp_path / "app.db"
    service = IngestionService(SQLiteCandidateRepository(path))
    pending = service.propose(document=_document(_structure(merged=True)), requester=REQUESTER)
    candidate = service.approve(pending.candidate_id, decided_by=REQUESTER)
    runs = SQLiteIndexRunRepository(path)
    runs.succeed(runs.create_running(candidate.candidate_id).id, added=1)
    clear_storage_engine_cache()
    repository = SQLiteCandidateRepository(path)
    assert repository.get(candidate.candidate_id) == candidate
    [restored] = repository.load_latest_successfully_indexed()
    assert restored.document.structured_content == candidate.document.structured_content
    assert candidate_to_langchain_documents(restored, chunk_size=1000, chunk_overlap=0)[0].page_content == candidate.document.content
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {"document_structure", "table_cells", "parsed_documents"} & tables


def test_legacy_sqlite_candidate_without_structure_uses_exact_old_identity(tmp_path):
    path = tmp_path / "legacy.db"
    candidate = IngestionService(SQLiteCandidateRepository(path)).propose(document=_document(), requester=REQUESTER)
    payload = candidate.model_dump(mode="json")
    for key in ("structured_content", "structure_hash"):
        payload["document"].pop(key)
    assert payload["document"]["schema_version"] == 1
    old_digest = hashlib.sha256("\x1f".join([
        "ingestion-candidate:v1", candidate.document.source_id, candidate.document.content_hash,
    ]).encode()).hexdigest()
    assert candidate.candidate_id == "cand_" + old_digest
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ingestion_candidates SET candidate_json = ?", (json.dumps(payload),))
    clear_storage_engine_cache()
    restored = SQLiteCandidateRepository(path).get(candidate.candidate_id)
    assert restored == candidate
    assert restored.document.structured_content is None
    assert restored.document.structure_hash is None


@pytest.mark.parametrize("damage", ["hash", "content", "structure", "missing_structure", "missing_hash"])
def test_structured_identity_or_projection_tampering_fails_closed(tmp_path, damage):
    path = tmp_path / "app.db"
    candidate = IngestionService(SQLiteCandidateRepository(path)).propose(
        document=_document(_structure()), requester=REQUESTER,
    )
    payload = candidate.model_dump(mode="json")
    document = payload["document"]
    if damage == "hash":
        document["structure_hash"] = "0" * 64
    elif damage == "content":
        document["content"] = "other"
        document["content_hash"] = build_content_hash("other")
    elif damage == "structure":
        document["structured_content"]["blocks"][0]["cells"][0]["is_header"] = True
    elif damage == "missing_structure":
        document.pop("structured_content")
    else:
        document.pop("structure_hash")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ingestion_candidates SET candidate_json = ?", (json.dumps(payload),))
    with pytest.raises(CandidatePersistenceError):
        SQLiteCandidateRepository(path).get(candidate.candidate_id)


@pytest.mark.parametrize("damage", ["object", "extra", "span", "overlap", "grid", "empty_grid", "order", "version", "id"])
def test_structure_rejects_unsafe_or_invalid_contract(damage):
    payload = _structure().model_dump(mode="json")
    block = payload["blocks"][0]
    if damage == "object":
        block["cells"][0]["text"] = object()
    elif damage == "extra":
        block["secret"] = "unvalidated-data"
    elif damage == "span":
        block["cells"][0]["row_span"] = 2
    elif damage == "overlap":
        block["cells"][0]["column_span"] = 2
    elif damage == "grid":
        block.update(row_count=1000, column_count=1000)
    elif damage == "empty_grid":
        block.update(row_count=100_000, column_count=0, cells=[])
    elif damage == "order":
        block["order"] = 3
    elif damage == "version":
        payload["structure_version"] = 2
    else:
        block["block_id"] = "api_key=unsafe"
    with pytest.raises(ValidationError):
        prepare_structured_content(payload)


def test_serialized_structure_and_block_count_limits(monkeypatch):
    with pytest.raises(ValidationError):
        ParsedDocument(blocks=tuple(TextBlock(block_id=f"b{i}", order=i, text="") for i in range(10_001)))
    monkeypatch.setattr("app.ingestion.document_blocks.STRUCTURE_MAX_SERIALIZED_BYTES", 100)
    with pytest.raises(ValidationError, match="大小限制"):
        _document(_structure())
