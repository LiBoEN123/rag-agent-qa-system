"""Evaluation manifest 可复现性与敏感字段测试。"""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.config import AppSettings
from scripts.evaluation import manifest as evaluation_manifest


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        _env_file=None,
        index_store_dir=tmp_path / "manifest",
        llm_api_key=SecretStr("do-not-leak"),
        query_rewrite_api_key=SecretStr("rewrite-do-not-leak"),
    )


def _coverage():
    return {
        "active_cases": 2,
        "suite": {"core": 2},
        "category": {"test": 2},
    }


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"x":1}', encoding="utf-8")

    assert evaluation_manifest.sha256_file(path) == (
        "5041bf1f713df204784353e82f6a4a5359"
        "31cb64f1f4b4a5aeaffcb720918b22"
    )


def test_build_manifest_records_commit_dirty_hashes_and_safe_whitelist(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    cases = tmp_path / "cases.json"
    rag = tmp_path / "rag.json"
    cases.write_text("[]", encoding="utf-8")
    rag.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_manifest,
        "get_git_state",
        lambda: {
            "git_commit": "abc123",
            "git_branch": "feature",
            "git_dirty": True,
        },
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "build_tool_contract",
        lambda _settings: (
            ["simple_calculator"],
            [{"name": "simple_calculator"}],
            "contract-hash",
        ),
    )

    manifest = evaluation_manifest.build_manifest(
        run_id="run-1",
        settings=settings,
        production_mcp_tools_enabled=True,
        agent_cases_path=cases,
        rag_eval_path=rag,
        coverage_report=_coverage(),
        top_k=5,
        decompose_top_k=10,
        include_online=False,
        use_llm_judge=False,
        judge_model="",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    serialized = json.dumps(manifest, ensure_ascii=False)

    assert manifest["code"]["git_commit"] == "abc123"
    assert manifest["code"]["git_dirty"] is True
    assert manifest["data"]["agent_cases_sha256"]
    assert manifest["agent"]["tool_contract_sha256"] == "contract-hash"
    assert manifest["mcp"] == {
        "production_enabled": True,
        "evaluation_enabled": False,
        "mode": "core_disabled",
    }
    assert manifest["rag"]["index_manifest_sha256"] == ""
    assert manifest["rag"]["prediction_schema_version"] == 2
    assert manifest["rag"]["max_context_chars"] == settings.max_context_chars
    assert manifest["rag"]["requested_top_k"] == 5
    assert manifest["rag"]["effective_top_k"] == 5
    assert manifest["rag"]["requested_decompose_top_k"] == 10
    assert manifest["rag"]["effective_decompose_top_k"] == 10
    assert (
        manifest["rag"]["context_selection_source"]
        == "app/rag/generation.py"
    )
    assert manifest["rag"]["context_selection_source_sha256"]
    assert "do-not-leak" not in serialized
    assert "rewrite-do-not-leak" not in serialized


def test_manifest_reads_index_manifest_without_loading_rag(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path)
    settings.index_store_dir.mkdir(parents=True)
    index_manifest = settings.index_store_dir / "index_manifest.json"
    index_manifest.write_text(
        json.dumps(
            {
                "embedding_model": "embedding-a",
                "rerank_model": "rerank-a",
                "build_config": {
                    "chunk_size": 123,
                    "chunk_overlap": 12,
                },
                "files": [{"path": "doc-a"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "get_git_state",
        lambda: {
            "git_commit": "abc",
            "git_branch": "branch",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "build_tool_contract",
        lambda _settings: ([], [], "hash"),
    )

    manifest = evaluation_manifest.build_manifest(
        run_id="run",
        settings=settings,
        production_mcp_tools_enabled=False,
        agent_cases_path=None,
        rag_eval_path=None,
        coverage_report=_coverage(),
        top_k=5,
        decompose_top_k=10,
        include_online=False,
        use_llm_judge=False,
        judge_model="",
        rag_status={
            "index_source": "chroma",
            "doc_count": 14,
            "chunk_count": 99,
        },
    )

    assert manifest["rag"]["index_source"] == "chroma"
    assert manifest["rag"]["document_count"] == 14
    assert manifest["rag"]["chunk_count"] == 99
    assert manifest["rag"]["chunk_size"] == 123


def test_manifest_mcp_metadata_does_not_read_or_expose_server_secrets(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={
            "mcp_servers_config_path": tmp_path / "mcp-with-secrets.json",
            "mcp_tools_enabled": False,
        }
    )
    settings.mcp_servers_config_path.write_text(
        json.dumps(
            {
                "headers": {
                    "Authorization": "Bearer manifest-must-not-read-this",
                    "X-Api-Key": "manifest-must-not-read-api-key",
                },
                "headers_env": {
                    "Authorization": "MCP_MANIFEST_SECRET_ENV"
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "MCP_MANIFEST_SECRET_ENV",
        "Bearer manifest-must-not-read-env-value",
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "get_git_state",
        lambda: {
            "git_commit": "abc",
            "git_branch": "branch",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "build_tool_contract",
        lambda _settings: ([], [], "hash"),
    )

    manifest = evaluation_manifest.build_manifest(
        run_id="mcp-safe",
        settings=settings,
        production_mcp_tools_enabled=True,
        agent_cases_path=None,
        rag_eval_path=None,
        coverage_report=_coverage(),
        top_k=5,
        decompose_top_k=10,
        include_online=False,
        use_llm_judge=False,
        judge_model="",
    )
    serialized = json.dumps(manifest, ensure_ascii=False)

    assert manifest["mcp"] == {
        "production_enabled": True,
        "evaluation_enabled": False,
        "mode": "core_disabled",
    }
    assert "manifest-must-not-read" not in serialized
    assert "MCP_MANIFEST_SECRET_ENV" not in serialized


@pytest.mark.parametrize(
    "value",
    [
        {"api_key": "x"},
        {"Authorization": "x"},
        {"nested": {"cookie": "x"}},
        {"value": SecretStr("x")},
    ],
)
def test_manifest_safety_rejects_sensitive_fields(value):
    with pytest.raises(ValueError):
        evaluation_manifest.assert_manifest_safe(value)


def test_static_tool_names_match_runtime_registry(tmp_path):
    settings = _settings(tmp_path)

    static_names = evaluation_manifest.read_registered_tool_names()
    runtime_names, _, contract_hash = (
        evaluation_manifest.build_tool_contract(settings)
    )

    assert static_names == runtime_names
    assert contract_hash


def test_extended_manifest_records_only_safe_mcp_catalog_metadata(
    tmp_path,
    monkeypatch,
):
    settings = _settings(tmp_path).model_copy(
        update={"mcp_tools_enabled": True}
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "get_git_state",
        lambda: {
            "git_commit": "abc",
            "git_branch": "extended",
            "git_dirty": True,
        },
    )
    monkeypatch.setattr(
        evaluation_manifest,
        "build_tool_contract",
        lambda _settings: ([], [], "hash"),
    )
    catalog = [
        {
            "server": "github",
            "tool_name": "github_get_file_contents",
            "remote_tool": "get_file_contents",
            "risk_level": "read_only",
            "approval_required": False,
            "enabled": True,
            "enabled_for_default_principal": True,
        }
    ]

    manifest = evaluation_manifest.build_manifest(
        run_id="extended",
        settings=settings,
        production_mcp_tools_enabled=True,
        agent_cases_path=None,
        rag_eval_path=None,
        coverage_report=_coverage(),
        top_k=5,
        decompose_top_k=10,
        include_online=False,
        use_llm_judge=False,
        judge_model="",
        selected_suites=["mcp", "mixed"],
        mcp_catalog=catalog,
        selected_mcp_servers=["github"],
    )

    assert manifest["evaluation_type"] == "extended_tool_evaluation"
    assert manifest["external_network_enabled"] is True
    assert manifest["web"] == {"enabled": True}
    assert manifest["mcp"] == {
        "production_enabled": True,
        "evaluation_enabled": True,
        "mode": "extended_live",
        "discovered_server_count": 1,
        "discovered_tool_count": 1,
        "selected_servers": ["github"],
        "tools": catalog,
    }
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "Authorization" not in serialized
    assert "do-not-leak" not in serialized
