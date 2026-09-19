"""Keep the public scripts surface intentionally small."""

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REMOVED = {
    "agent_eval.py",
    "predictions.py",
    "evaluate_rag_results_semantic_fixed.py",
    "perf_test.py",
    "concurrent_perf_test.py",
    "test_local_llm.py",
    "smoke_test_web_search.py",
    "smoke_test_web_fetch.py",
    "smoke_test_web_pdf_fetch.py",
    "generate_web_pdf_fixture.py",
}
PUBLIC_CLIS = {
    "run_evaluation_baseline.py",
    "compare_evaluation_runs.py",
    "benchmark_chat_api.py",
    "smoke_test_online_tools.py",
    "smoke_test_agent_tool_calling.py",
    "smoke_test_chat_stream.py",
    "smoke_test_mcp_tools.py",
    "inspect_agent_runtime.py",
    "cache_models.py",
    "summarize_logs.py",
}


def test_removed_scripts_are_absent_and_root_contains_only_public_clis():
    assert all(not (SCRIPTS_DIR / name).exists() for name in REMOVED)
    assert {path.name for path in SCRIPTS_DIR.glob("*.py")} == PUBLIC_CLIS


def test_evaluation_internal_modules_import_without_legacy_cli_paths():
    modules = [
        "scripts.evaluation.agent",
        "scripts.evaluation.comparison",
        "scripts.evaluation.contracts",
        "scripts.evaluation.manifest",
        "scripts.evaluation.rag_predictions",
        "scripts.evaluation.rag_evaluator",
        "scripts.evaluation.report",
        "scripts.evaluation.runner",
        "scripts.evaluation.rag.common",
        "scripts.evaluation.rag.evaluation",
        "scripts.evaluation.rag.judge",
        "scripts.evaluation.rag.report",
        "scripts.evaluation.rag.retrieval",
    ]

    assert [importlib.import_module(name).__name__ for name in modules] == modules
