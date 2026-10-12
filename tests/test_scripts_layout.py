"""Keep the public scripts surface intentionally small."""

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PUBLIC_CLIS = {
    "run_evaluation_baseline.py",
    "compare_evaluation_runs.py",
    "smoke_test_mcp_tools.py",
    "inspect_agent_runtime.py",
}


def test_scripts_root_contains_public_clis():
    assert {path.name for path in SCRIPTS_DIR.glob("*.py")} == PUBLIC_CLIS


def test_evaluation_internal_modules_are_importable():
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
        "scripts.evaluation.deepeval_adapter",
        "scripts.evaluation.rag.retrieval",
    ]

    assert [importlib.import_module(name).__name__ for name in modules] == modules
