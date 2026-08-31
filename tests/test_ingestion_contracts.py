"""Phase 9 KnowledgeDocument identity 与 serialization contracts。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.ingestion import (
    IngestionPrincipalType,
    IngestionRequester,
    IngestionService,
    KnowledgeDocument,
    KnowledgeSourceType,
    build_knowledge_document,
)


FIXED_TIME = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _document(
    *,
    source_type=KnowledgeSourceType.WEB,
    source_uri="https://example.com/guide",
    content="stable content",
    version=None,
    metadata=None,
):
    return build_knowledge_document(
        source_type=source_type,
        source_uri=source_uri,
        title="Guide",
        content=content,
        version=version,
        fetched_at=FIXED_TIME,
        metadata=metadata,
    )


def _requester() -> IngestionRequester:
    return IngestionRequester(
        principal_id="phase9-test-user",
        principal_type=IngestionPrincipalType.USER,
        source="pytest",
    )


def test_same_source_and_normalized_content_have_stable_ids():
    first = _document(content="Cafe\u0301\r\nline  ")
    second = _document(content="Café\nline")
    first_candidate = IngestionService().propose(
        document=first,
        requester=_requester(),
        requested_at=FIXED_TIME,
    )
    second_candidate = IngestionService().propose(
        document=second,
        requester=_requester(),
        requested_at=FIXED_TIME,
    )

    assert first.content == second.content == "Café\nline"
    assert first.source_id == second.source_id
    assert first.document_uid == second.document_uid
    assert first.content_hash == second.content_hash
    assert first_candidate.candidate_id == second_candidate.candidate_id


def test_same_source_changed_content_keeps_source_but_changes_candidate():
    first = _document(content="content A")
    second = _document(content="content B")
    service = IngestionService()

    first_candidate = service.propose(document=first, requester=_requester())
    second_candidate = service.propose(document=second, requester=_requester())

    assert first.source_id == second.source_id
    assert first.document_uid == second.document_uid
    assert first.content_hash != second.content_hash
    assert first_candidate.candidate_id != second_candidate.candidate_id


def test_different_source_same_content_preserves_provenance_identity():
    first = _document(source_uri="https://example.com/a", content="same")
    second = _document(source_uri="https://example.com/b", content="same")
    first_candidate = IngestionService().propose(
        document=first,
        requester=_requester(),
    )
    second_candidate = IngestionService().propose(
        document=second,
        requester=_requester(),
    )

    assert first.content_hash == second.content_hash
    assert first.source_id != second.source_id
    assert first.document_uid != second.document_uid
    assert first_candidate.candidate_id != second_candidate.candidate_id


def test_github_commit_change_without_content_change_keeps_content_identity():
    first = _document(
        source_type=KnowledgeSourceType.GITHUB,
        source_uri="github://LiBoEN123/rag-agent-qa-system/app/main.py",
        content="same file",
        version="commit-a",
        metadata={"commit_sha": "commit-a", "path": "app/main.py"},
    )
    second = _document(
        source_type=KnowledgeSourceType.GITHUB,
        source_uri="github://liboen123/RAG-Agent-QA-System/app/main.py",
        content="same file",
        version="commit-b",
        metadata={"commit_sha": "commit-b", "path": "app/main.py"},
    )

    assert first.source_id == second.source_id
    assert first.document_uid == second.document_uid
    assert first.content_hash == second.content_hash
    assert first.version != second.version


def test_github_content_change_changes_candidate_not_source_identity():
    first = _document(
        source_type="github",
        source_uri="github://owner/repo/docs/guide.md",
        content="old",
        version="commit-a",
    )
    second = _document(
        source_type="github",
        source_uri="github://owner/repo/docs/guide.md",
        content="new",
        version="commit-b",
    )
    service = IngestionService()

    first_candidate = service.propose(document=first, requester=_requester())
    second_candidate = service.propose(document=second, requester=_requester())

    assert first.source_id == second.source_id
    assert first.content_hash != second.content_hash
    assert first_candidate.candidate_id != second_candidate.candidate_id


def test_canonical_web_url_normalizes_case_and_fragment_but_preserves_query():
    uppercase = _document(source_uri="HTTPS://Example.COM/path?page=2#fragment")
    lowercase = _document(source_uri="https://example.com/path?page=2")
    different_query = _document(source_uri="https://example.com/path?page=3")

    assert uppercase.source_uri == "https://example.com/path?page=2"
    assert uppercase.source_id == lowercase.source_id
    assert uppercase.source_id != different_query.source_id


def test_contracts_survive_json_serialization_and_reload():
    document = _document(metadata={"nested": {"values": [1, True, None]}})
    candidate = IngestionService().propose(
        document=document,
        requester=_requester(),
        requested_at=FIXED_TIME,
    )

    document_payload = document.model_dump(mode="json")
    candidate_payload = candidate.model_dump(mode="json")
    assert json.dumps(document_payload, allow_nan=False)
    assert json.dumps(candidate_payload, allow_nan=False)
    assert KnowledgeDocument.model_validate(document_payload) == document
    assert type(candidate).model_validate(candidate_payload) == candidate
    assert document_payload["schema_version"] == 1
    assert candidate_payload["schema_version"] == 1


def test_stable_ids_do_not_depend_on_python_hash_randomization():
    script = """
from datetime import datetime, timezone
from app.ingestion import IngestionPrincipalType, IngestionRequester, IngestionService, build_knowledge_document
doc = build_knowledge_document(source_type='web', source_uri='https://example.com/path', title='x', content='same', fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
requester = IngestionRequester(principal_id='subprocess-user', principal_type=IngestionPrincipalType.USER, source='pytest')
candidate = IngestionService().propose(document=doc, requester=requester)
print(doc.source_id, doc.document_uid, doc.content_hash, candidate.candidate_id)
"""
    outputs = []
    for seed in ("1", "999"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            check=True,
            text=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]


def test_unknown_source_type_and_extra_fields_are_rejected():
    with pytest.raises(ValueError, match="KnowledgeSourceType"):
        _document(source_type="unknown")

    payload = _document().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        KnowledgeDocument.model_validate(payload)


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"object": object()},
        {"bytes": b"secret"},
        {"secret": SecretStr("hidden")},
        {"nan": float("nan")},
        {"tuple": ("not", "json")},
    ],
)
def test_non_json_safe_metadata_is_rejected(invalid_metadata):
    with pytest.raises((TypeError, ValueError)):
        _document(metadata=invalid_metadata)


def test_factory_does_not_treat_an_invalid_empty_metadata_container_as_missing():
    with pytest.raises(TypeError, match="metadata 必须是 dict"):
        _document(metadata=[])
