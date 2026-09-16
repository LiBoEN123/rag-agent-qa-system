"""框架无关日志与 Uvicorn HTTP 日志适配的边界测试。"""

import asyncio
import importlib
import logging

import app.api.application as api_application
from app.api.http_logging import (
    DecodedUvicornAccessFilter,
    configure_uvicorn_access_log,
)
from app.common import logging as common_logging


def _access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:50000", "GET", path, "1.1", 200),
        exc_info=None,
    )


def test_uvicorn_filter_decodes_url_and_escapes_control_characters():
    record = _access_record("/%E4%B8%AD%E6%96%87%0Aunsafe")

    assert DecodedUvicornAccessFilter().filter(record) is True

    assert record.args == (
        "127.0.0.1:50000",
        "GET",
        "/中文\\x0aunsafe",
        "1.1",
        200,
    )


def test_common_logging_import_does_not_configure_uvicorn_access():
    access_logger = logging.getLogger("uvicorn.access")
    original_filters = list(access_logger.filters)
    access_logger.filters[:] = []
    try:
        importlib.reload(common_logging)

        assert access_logger.filters == []
    finally:
        access_logger.filters[:] = original_filters


def test_uvicorn_access_configuration_is_explicit_and_idempotent():
    access_logger = logging.getLogger("uvicorn.access")
    original_filters = list(access_logger.filters)
    access_logger.filters[:] = []
    try:
        assert access_logger.filters == []

        configure_uvicorn_access_log()
        configure_uvicorn_access_log()

        matching_filters = [
            log_filter
            for log_filter in access_logger.filters
            if getattr(log_filter, "marker", "")
            == DecodedUvicornAccessFilter.marker
        ]
        assert len(matching_filters) == 1
    finally:
        access_logger.filters[:] = original_filters


def test_fastapi_lifespan_explicitly_configures_uvicorn_access(monkeypatch):
    calls = []

    async def close_agent():
        calls.append("close_agent")

    monkeypatch.setattr(
        api_application,
        "configure_uvicorn_access_log",
        lambda: calls.append("configure_http_logging"),
    )
    monkeypatch.setattr(
        api_application,
        "start_application_resources",
        lambda *_args: calls.append("start"),
    )
    monkeypatch.setattr(
        api_application,
        "close_tool_executor",
        lambda: calls.append("close_tools"),
    )
    monkeypatch.setattr(
        api_application,
        "close_sql_engine_runtime",
        lambda: calls.append("close_sql_engines"),
    )
    monkeypatch.setattr(
        api_application,
        "aclose_agent_runtime",
        close_agent,
    )

    async def run_lifespan():
        application = api_application.create_app()
        async with api_application.lifespan(application):
            assert calls == ["configure_http_logging", "start"]

    asyncio.run(run_lifespan())

    assert calls == [
        "configure_http_logging",
        "start",
        "close_tools",
        "close_sql_engines",
        "close_agent",
    ]
