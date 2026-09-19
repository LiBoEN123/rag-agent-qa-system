"""配置及主要消费者的导入副作用回归测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import AppSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_TARGETS = (
    "app.config",
    "app.tools.catalog",
    "app.llm.factory",
    "app.agent.runtime",
    "app.services.chat_service",
    "app.api.chat",
    "app.api.application",
    "app.main",
    "scripts.inspect_agent_runtime",
)


def _clean_subprocess_environment() -> dict[str, str]:
    """移除应用配置，保留解释器运行所需的宿主环境。"""
    application_names = {
        name.upper() for name in AppSettings.model_fields
    } | {
        "MODEL_CACHE_DIR",
        "DEEPSEEK_API_KEY",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in application_names
    }
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    return environment


def _run_import_probe(tmp_path: Path, source: str) -> subprocess.CompletedProcess:
    """在隔离目录和清洁环境中执行导入探针子进程。"""
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=_clean_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_import_app_config_does_not_construct_or_read_settings(tmp_path):
    """导入配置模块不实例化 BaseSettings，也不读取当前目录的 .env。"""
    (tmp_path / ".env").write_text(
        "TOP_K=not-an-integer\nLLM_API_KEY=local-secret\n",
        encoding="utf-8",
    )
    source = """
import dotenv
from pydantic_settings import BaseSettings

def forbidden(*args, **kwargs):
    raise AssertionError("settings or dotenv was evaluated during import")

BaseSettings.__init__ = forbidden
dotenv.load_dotenv = forbidden

import app.config as config

assert config.get_settings.cache_info().currsize == 0
"""

    result = _run_import_probe(tmp_path, source)

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]


def test_primary_consumers_import_without_key_or_runtime_side_effects(tmp_path):
    """缺少 key 时主要消费者可导入，且所有资源缓存仍为空。"""
    (tmp_path / ".env").write_text(
        "TOP_K=not-an-integer\nLLM_API_KEY=local-secret\n",
        encoding="utf-8",
    )
    imports = "\n".join(
        f"import {module}" for module in IMPORT_TARGETS
    )
    source = f"""
import dotenv
import sqlite3
from pydantic_settings import BaseSettings

def forbidden(*args, **kwargs):
    raise AssertionError("external resource or settings created during import")

BaseSettings.__init__ = forbidden
dotenv.load_dotenv = forbidden
sqlite3.connect = forbidden

{imports}

from app import config
from app.agent import runtime as agent_runtime
from app.llm import factory
import app.tools.catalog as catalog
import app.tools.runtime as runtime

assert config.get_settings.cache_info().currsize == 0
assert catalog.get_local_tool_specs.cache_info().currsize == 0
from app.agent.framework_tools import get_local_tools
assert get_local_tools.cache_info().currsize == 0
assert agent_runtime._get_cached_framework_agent.cache_info().currsize == 0
assert runtime._executor is None
assert not hasattr(runtime, "_runtime_entries")
assert all(
    getter.cache_info().currsize == 0
    for getter in (
        factory.get_answer_model,
        factory.get_query_rewrite_model,
        factory.get_judge_model,
    )
)
"""

    result = _run_import_probe(tmp_path, source)

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == [".env"]
