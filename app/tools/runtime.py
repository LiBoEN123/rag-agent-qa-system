"""LangChain 尚未提供的同步工具超时与线程池生命周期。"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from typing import Any


logger = logging.getLogger(__name__)


class ProjectToolTimeoutError(Exception):
    """同步 handler 超过 ToolSpec timeout。"""


class ProjectToolExecutorError(Exception):
    """共享 timeout executor 无法接受或完成任务。"""


_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_executor_workers = 0


def _shared_executor(max_workers: int) -> ThreadPoolExecutor:
    """按当前配置惰性创建唯一共享有界 executor。"""
    global _executor, _executor_workers
    with _executor_lock:
        if _executor is not None and _executor_workers == max_workers:
            return _executor
        previous = _executor
        _executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tool-timeout",
        )
        _executor_workers = max_workers
    if previous is not None:
        previous.shutdown(wait=False, cancel_futures=True)
    return _executor


def run_with_timeout(
    function: Callable[[], Any],
    *,
    timeout: float,
    max_workers: int,
) -> Any:
    """在线程池执行同步 handler，并只负责调用方等待超时。"""
    try:
        future = _shared_executor(max_workers).submit(function)
    except Exception as exc:
        raise ProjectToolExecutorError from exc
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        if future.done():
            raise
        future.cancel()
        raise ProjectToolTimeoutError from exc
    except CancelledError as exc:
        raise ProjectToolExecutorError from exc


def close_tool_executor() -> None:
    """关闭唯一共享 executor；不等待已经开始运行的 handler。"""
    global _executor, _executor_workers
    with _executor_lock:
        executor = _executor
        _executor = None
        _executor_workers = 0
    if executor is None:
        return
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        logger.warning("tool executor shutdown failed: %s", type(exc).__name__)


__all__ = [
    "ProjectToolExecutorError",
    "ProjectToolTimeoutError",
    "close_tool_executor",
    "run_with_timeout",
]
