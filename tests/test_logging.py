"""框架无关日志与 Uvicorn HTTP 日志适配的边界测试。"""

import asyncio
import importlib
import logging

import app.api.chat as chat_api
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

    monkeypatch.setattr(
        chat_api,
        "configure_uvicorn_access_log",
        lambda: calls.append("configure_http_logging"),
    )
    monkeypatch.setattr(
        chat_api,
        "start_on_event",
        lambda: calls.append("start"),
    )
    monkeypatch.setattr(
        chat_api,
        "close_tool_runtime",
        lambda: calls.append("close_tools"),
    )
    monkeypatch.setattr(
        chat_api,
        "close_agent_graph_runtime",
        lambda: calls.append("close_graph"),
    )

    async def run_lifespan():
        async with chat_api.lifespan(chat_api.app):
            assert calls == ["configure_http_logging", "start"]

    asyncio.run(run_lifespan())

    assert calls == [
        "configure_http_logging",
        "start",
        "close_tools",
        "close_graph",
    ]
