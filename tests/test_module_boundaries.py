"""Import Linter 是模块依赖方向的唯一图分析器。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_LINTER_CONFIG = PROJECT_ROOT / ".importlinter"


def test_import_linter_configuration_has_all_boundaries_and_no_ignores():
    config = IMPORT_LINTER_CONFIG.read_text(encoding="utf-8")

    assert "ignore_imports" not in config
    assert "contract:application-layers" in config
    assert "contract:top-level-acyclic" in config
    assert "contract:foundation-does-not-depend-on-application" in config
    assert "contract:ingestion-domain-is-independent" in config
    assert "contract:http-framework-only-in-api" in config
    assert "contract:pydantic-settings-only-in-config" in config
    for name in (
        "rag-read-side-independence", "indexing-write-side-independence",
        "sources-do-not-index", "agent-no-ingestion-side-effects", "chat-storage-is-independent",
    ):
        assert f"contract:{name}" in config
