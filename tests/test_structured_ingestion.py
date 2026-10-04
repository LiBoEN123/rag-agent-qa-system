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
from app.indexing.langchain import candidate_to_langchain_documents
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
        TextBlock(block_id="b0", order=0, text="Cafe\u0301\r\nAuthorization: Bearer block-secret  ",
                  text_role="section_header", heading_level=2),
        TableBlock(block_id="b1", order=1, caption="api_key=caption-secret", row_count=1, column_count=1,
                   cells=(TableCell(text="password=cell-secret\r\ntrail  ", row_index=0, column_index=0),)),
    ))
    document = _document(structure)
    assert document.content_redacted
    assert document.structured_content.blocks[0].text.startswith("Café\n")
    assert document.structured_content.blocks[0].text_role == "section_header"
    assert document.structured_content.blocks[0].heading_level == 2
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
    runs.succeed(runs.create_running(candidate.candidate_id, source_id=candidate.document.source_id).id, added=1)
    clear_storage_engine_cache()
    repository = SQLiteCandidateRepository(path)
    assert repository.get(candidate.candidate_id) == candidate
    [restored] = repository.load_latest_successfully_indexed()
    assert restored.document.structured_content == candidate.document.structured_content
    [chunk] = candidate_to_langchain_documents(restored, chunk_size=1000, chunk_overlap=0)
    assert chunk.page_content == "[标题路径] table\n\n[表格]\n表头：列1；列2\n说明：value\n[/表格]"
    assert chunk.metadata["element_type"] == "table"
    # 全宽合并说明行不是 data row；结构本身仍完整保留在 candidate。
    assert "row_start" not in chunk.metadata and "row_end" not in chunk.metadata
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


def test_legacy_structured_candidate_keeps_hash_identity_and_sqlite_reload(tmp_path):
    # 精确旧表示：不调用新 model_dump 生成 legacy hash。
    old_structure = {"structure_version": 1, "page_count": None, "blocks": [{
        "kind": "text", "block_id": "b0", "order": 0, "page_number": None, "text": "旧正文",
    }]}
    old_hash = hashlib.sha256(json.dumps(
        old_structure, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    path = tmp_path / "legacy-structured.db"
    service = IngestionService(SQLiteCandidateRepository(path))
    candidate = service.propose(document=_document(ParsedDocument.model_validate(old_structure)), requester=REQUESTER)
    old_candidate_id = "cand_" + hashlib.sha256("\x1f".join([
        "ingestion-candidate:structured-v1", candidate.document.source_id,
        candidate.document.content_hash, old_hash,
    ]).encode("utf-8")).hexdigest()
    assert candidate.document.structure_hash == old_hash
    assert candidate.candidate_id == old_candidate_id
    payload = candidate.model_dump(mode="json")
    payload["document"].update(structured_content=old_structure, structure_hash=old_hash)
    assert KnowledgeDocument.model_validate(payload["document"]) == candidate.document
    assert IngestionCandidate.model_validate(payload) == candidate
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE ingestion_candidates SET candidate_json = ?", (json.dumps(payload),))
    clear_storage_engine_cache()
    restored = SQLiteCandidateRepository(path).get(old_candidate_id)
    assert restored == candidate
    assert restored.document.structured_content.blocks[0].text_role == "text"
    assert restored.document.structured_content.blocks[0].heading_level is None
    assert build_structure_hash(restored.document.structured_content) == old_hash
    assert IngestionCandidate.model_validate_json(restored.model_dump_json()) == restored


def test_heading_role_and_level_change_identity_and_tampering_is_rejected():
    documents = [_document(ParsedDocument(blocks=(TextBlock(
        block_id="b0", order=0, text="相同标题", text_role=role, heading_level=level,
    ),))) for role, level in (("text", None), ("title", None), ("section_header", 1), ("section_header", 7))]
    assert len({document.content_hash for document in documents}) == 1
    assert len({document.structure_hash for document in documents}) == 4
    service = IngestionService()
    assert len({service.propose(document=document, requester=REQUESTER).candidate_id for document in documents}) == 4
    payload = documents[-1].model_dump(mode="json")
    payload["structured_content"]["blocks"][0]["heading_level"] = 2
    with pytest.raises(ValidationError, match="structure_hash"):
        KnowledgeDocument.model_validate(payload)


@pytest.mark.parametrize("role, level", [
    ("text", 1), ("title", 1), ("section_header", None), ("section_header", 0),
    ("section_header", True), ("section_header", "2"),
])
def test_heading_contract_rejects_invalid_role_level_pairs(role, level):
    with pytest.raises(ValidationError):
        TextBlock(block_id="b0", order=0, text="标题", text_role=role, heading_level=level)


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
