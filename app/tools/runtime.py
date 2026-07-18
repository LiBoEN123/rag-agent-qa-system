"""统一工具运行时。

负责工具查找、参数校验、受限线程池执行、超时、有限重试、结果校验和
运行元数据记录。同步公共入口保持不变。
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from pydantic import ValidationError

from app.config import Settings, validate_tool_runtime_config
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.result import tool_error

if TYPE_CHECKING:
    from app.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

_RuntimeConfig = tuple[int, int, int, float]


@dataclass(frozen=True, slots=True)
class _RuntimeEntry:
    """记录某个注册表当前绑定的 Runtime 及其配置。"""

    runtime: ToolRuntime
    config: _RuntimeConfig


class ToolRuntime:
    """使用共享且有界的线程池同步执行注册工具。"""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_workers: int,
        retry_base_delay_ms: int,
        retry_max_delay_ms: int,
        retry_jitter_ratio: float,
        sleep_func: Callable[[float], None] = time.sleep,
        random_func: Callable[[float, float], float] = random.uniform,
    ) -> None:
        """创建 Runtime；线程池只创建一次并在后续调用间复用。"""
        validate_tool_runtime_config(
            max_workers=max_workers,
            retry_base_delay_ms=retry_base_delay_ms,
            retry_max_delay_ms=retry_max_delay_ms,
            retry_jitter_ratio=retry_jitter_ratio,
        )
        self.registry = registry
        self._retry_base_delay = retry_base_delay_ms / 1000.0
        self._retry_max_delay = retry_max_delay_ms / 1000.0
        self._retry_jitter_ratio = float(retry_jitter_ratio)
        self._sleep_func = sleep_func
        self._random_func = random_func
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tool-runtime",
        )
        self._state_lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """返回 Runtime 是否已经关闭。"""
        with self._state_lock:
            return self._closed

    def execute(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        *,
        context: ToolInvocationContext | None = None,
    ) -> dict:
        """同步执行工具，并把所有预期失败转换为统一结果字典。"""
        started_at = time.perf_counter()
        raw_input: object = {} if tool_args is None else tool_args
        compatible_input = raw_input if isinstance(raw_input, dict) else {}
        attempts: list[dict[str, Any]] = []

        if self.closed:
            return self._error_result(
                tool_name=tool_name,
                tool_input=compatible_input,
                error="tool runtime is unavailable: runtime is closed",
                error_code=ToolErrorCode.RUNTIME_UNAVAILABLE,
                retryable=False,
                spec=None,
                attempt_count=0,
                max_attempts=0,
                timed_out=False,
                started_at=started_at,
                attempts=attempts,
            )

        spec = self.registry.get(tool_name)
        if spec is None:
            return self._error_result(
                tool_name=tool_name,
                tool_input=compatible_input,
                error=f"unknown tool: {tool_name}",
                error_code=ToolErrorCode.UNKNOWN_TOOL,
                retryable=False,
                spec=None,
                attempt_count=0,
                max_attempts=0,
                timed_out=False,
                started_at=started_at,
                attempts=attempts,
            )

        try:
            validated_args = spec.args_model.model_validate(raw_input).model_dump()
        except ValidationError as exc:
            return self._error_result(
                tool_name=spec.name,
                tool_input=compatible_input,
                error=f"invalid tool args: {exc}",
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
                retryable=False,
                spec=spec,
                attempt_count=0,
                max_attempts=0,
                timed_out=False,
                started_at=started_at,
                attempts=attempts,
            )

        max_attempts = self._max_attempts(spec)
        invocation_context = context or ToolInvocationContext()
        attempt_count = 0
        timed_out = False

        for attempt in range(1, max_attempts + 1):
            attempt_started_at = time.perf_counter()
            try:
                with self._state_lock:
                    if self._closed:
                        return self._error_result(
                            tool_name=spec.name,
                            tool_input=validated_args,
                            error=(
                                "tool runtime is unavailable: runtime is closed"
                            ),
                            error_code=ToolErrorCode.RUNTIME_UNAVAILABLE,
                            retryable=False,
                            spec=spec,
                            attempt_count=attempt_count,
                            max_attempts=max_attempts,
                            timed_out=timed_out,
                            started_at=started_at,
                            attempts=attempts,
                        )
                    future = self._executor.submit(
                        spec.handler,
                        validated_args,
                        invocation_context,
                    )
                attempt_count = attempt
            except Exception as exc:
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error=f"tool runtime is unavailable: {exc}",
                    error_code=ToolErrorCode.RUNTIME_UNAVAILABLE,
                    retryable=False,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=timed_out,
                    started_at=started_at,
                    attempts=attempts,
                )

            try:
                raw_result = future.result(timeout=spec.timeout)
            except FutureTimeoutError as exc:
                if future.done():
                    self._record_attempt(
                        attempts,
                        attempt=attempt,
                        started_at=attempt_started_at,
                        error_code=ToolErrorCode.EXECUTION_ERROR,
                        timed_out=False,
                    )
                    return self._error_result(
                        tool_name=spec.name,
                        tool_input=validated_args,
                        error=str(exc),
                        error_code=ToolErrorCode.EXECUTION_ERROR,
                        retryable=False,
                        spec=spec,
                        attempt_count=attempt_count,
                        max_attempts=max_attempts,
                        timed_out=timed_out,
                        started_at=started_at,
                        attempts=attempts,
                    )
                future.cancel()
                timed_out = True
                self._record_attempt(
                    attempts,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    error_code=ToolErrorCode.TIMEOUT,
                    timed_out=True,
                )
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error=(
                        f"tool {spec.name!r} timed out after "
                        f"{spec.timeout} seconds"
                    ),
                    error_code=ToolErrorCode.TIMEOUT,
                    retryable=True,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=True,
                    started_at=started_at,
                    attempts=attempts,
                )
            except ToolExecutionError as exc:
                error_code = exc.code
                retryable = exc.retryable
                timed_out = timed_out or error_code is ToolErrorCode.TIMEOUT
                self._record_attempt(
                    attempts,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    error_code=error_code,
                    timed_out=error_code is ToolErrorCode.TIMEOUT,
                )
                if self._should_retry(
                    spec=spec,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_code=error_code,
                    retryable=retryable,
                ):
                    self._sleep_before_retry(retry_index=attempt - 1)
                    continue
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error=str(exc),
                    error_code=error_code,
                    retryable=retryable,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=timed_out,
                    started_at=started_at,
                    attempts=attempts,
                )
            except CancelledError:
                self._record_attempt(
                    attempts,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    error_code=ToolErrorCode.RUNTIME_UNAVAILABLE,
                    timed_out=False,
                )
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error="tool runtime is unavailable: execution was cancelled",
                    error_code=ToolErrorCode.RUNTIME_UNAVAILABLE,
                    retryable=False,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=timed_out,
                    started_at=started_at,
                    attempts=attempts,
                )
            except Exception as exc:
                self._record_attempt(
                    attempts,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    error_code=ToolErrorCode.EXECUTION_ERROR,
                    timed_out=False,
                )
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error=str(exc),
                    error_code=ToolErrorCode.EXECUTION_ERROR,
                    retryable=False,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=timed_out,
                    started_at=started_at,
                    attempts=attempts,
                )

            normalized_result, error_code, error_message = (
                self._normalize_handler_result(
                    spec=spec,
                    validated_args=validated_args,
                    raw_result=raw_result,
                )
            )
            self._record_attempt(
                attempts,
                attempt=attempt,
                started_at=attempt_started_at,
                error_code=error_code,
                timed_out=False,
            )
            if error_code is not None:
                return self._error_result(
                    tool_name=spec.name,
                    tool_input=validated_args,
                    error=error_message,
                    error_code=error_code,
                    retryable=False,
                    spec=spec,
                    attempt_count=attempt_count,
                    max_attempts=max_attempts,
                    timed_out=timed_out,
                    started_at=started_at,
                    attempts=attempts,
                )

            assert normalized_result is not None
            normalized_result["runtime"] = self._runtime_metadata(
                spec=spec,
                attempt_count=attempt_count,
                max_attempts=max_attempts,
                timed_out=timed_out,
                started_at=started_at,
                attempts=attempts,
            )
            return normalized_result

        raise RuntimeError("tool runtime reached an unreachable execution state")

    def close(self) -> None:
        """幂等关闭线程池，不等待或强制终止已经运行的 handler。"""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.warning(
                "tool runtime shutdown failed: %s",
                str(exc)[:200],
            )

    @staticmethod
    def _max_attempts(spec: ToolSpec) -> int:
        """计算最大执行次数，并防御非幂等契约被绕过的情况。"""
        if not spec.idempotent:
            return 1
        if isinstance(spec.max_retries, bool) or not isinstance(
            spec.max_retries,
            int,
        ):
            return 1
        return 1 + max(spec.max_retries, 0)

    @staticmethod
    def _should_retry(
        *,
        spec: ToolSpec,
        attempt: int,
        max_attempts: int,
        error_code: ToolErrorCode,
        retryable: bool,
    ) -> bool:
        """只允许幂等工具对明确可重试且非超时的错误重试。"""
        return (
            attempt < max_attempts
            and spec.idempotent
            and retryable
            and error_code is not ToolErrorCode.TIMEOUT
        )

    def _sleep_before_retry(self, *, retry_index: int) -> None:
        """按重试序号执行带抖动的指数退避。"""
        delay = min(
            self._retry_base_delay * (2**retry_index),
            self._retry_max_delay,
        )
        delay *= self._random_func(
            1.0 - self._retry_jitter_ratio,
            1.0 + self._retry_jitter_ratio,
        )
        if delay > 0:
            self._sleep_func(delay)

    @staticmethod
    def _normalize_handler_result(
        *,
        spec: ToolSpec,
        validated_args: dict[str, Any],
        raw_result: object,
    ) -> tuple[dict[str, Any] | None, ToolErrorCode | None, str]:
        """校验并规范化 handler 返回结构和可选输出模型。"""
        if not isinstance(raw_result, dict):
            return (
                None,
                ToolErrorCode.INVALID_RESULT,
                "tool handler result must be a dict, got "
                f"{type(raw_result).__name__}",
            )

        status = raw_result.get("status")
        ok = raw_result.get("ok")
        is_success = status == "success" and ok is True
        is_error = status == "error" and ok is False
        if not is_success and not is_error:
            return (
                None,
                ToolErrorCode.INVALID_RESULT,
                "tool handler result has inconsistent status and ok fields",
            )

        if is_error:
            return (
                None,
                ToolErrorCode.EXECUTION_ERROR,
                str(raw_result.get("error") or "tool returned an error result"),
            )

        output = raw_result.get("output")
        if spec.output_model is not None:
            try:
                output = spec.output_model.model_validate(output).model_dump()
            except ValidationError as exc:
                return (
                    None,
                    ToolErrorCode.OUTPUT_VALIDATION_ERROR,
                    f"invalid tool output: {exc}",
                )

        normalized_result = dict(raw_result)
        normalized_result.update(
            {
                "status": "success",
                "ok": True,
                "tool_name": spec.name,
                "tool_input": validated_args,
                "output": output,
                "error": "",
                "error_code": "",
                "retryable": False,
            }
        )
        return normalized_result, None, ""

    @staticmethod
    def _record_attempt(
        attempts: list[dict[str, Any]],
        *,
        attempt: int,
        started_at: float,
        error_code: ToolErrorCode | None,
        timed_out: bool,
    ) -> None:
        """记录一次 handler 提交后的执行耗时和结果分类。"""
        attempts.append(
            {
                "attempt": attempt,
                "elapsed_ms": max(
                    0,
                    int((time.perf_counter() - started_at) * 1000),
                ),
                "error_code": error_code.value if error_code else "",
                "timed_out": timed_out,
            }
        )

    @staticmethod
    def _runtime_metadata(
        *,
        spec: ToolSpec | None,
        attempt_count: int,
        max_attempts: int,
        timed_out: bool,
        started_at: float,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构造兼容且可序列化的 Runtime 元数据。"""
        return {
            "tool_version": spec.version if spec is not None else "",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "elapsed_ms": max(
                0,
                int((time.perf_counter() - started_at) * 1000),
            ),
            "timeout": spec.timeout if spec is not None else None,
            "timed_out": timed_out,
            "attempts": [dict(item) for item in attempts],
        }

    def _error_result(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        error: str,
        error_code: ToolErrorCode,
        retryable: bool,
        spec: ToolSpec | None,
        attempt_count: int,
        max_attempts: int,
        timed_out: bool,
        started_at: float,
        attempts: list[dict[str, Any]],
    ) -> dict:
        """构造带完整 Runtime 元数据的失败结果。"""
        return tool_error(
            tool_name,
            tool_input,
            error,
            error_code=error_code.value,
            retryable=retryable,
            runtime=self._runtime_metadata(
                spec=spec,
                attempt_count=attempt_count,
                max_attempts=max_attempts,
                timed_out=timed_out,
                started_at=started_at,
                attempts=attempts,
            ),
        )


_runtime_lock = threading.Lock()
_runtime_entries: WeakKeyDictionary[ToolRegistry, _RuntimeEntry] = (
    WeakKeyDictionary()
)


def _settings_runtime_config() -> _RuntimeConfig:
    """读取并校验当前进程配置，供惰性 Runtime 初始化使用。"""
    validate_tool_runtime_config(
        max_workers=Settings.TOOL_RUNTIME_MAX_WORKERS,
        retry_base_delay_ms=Settings.TOOL_RETRY_BASE_DELAY_MS,
        retry_max_delay_ms=Settings.TOOL_RETRY_MAX_DELAY_MS,
        retry_jitter_ratio=Settings.TOOL_RETRY_JITTER_RATIO,
    )
    return (
        Settings.TOOL_RUNTIME_MAX_WORKERS,
        Settings.TOOL_RETRY_BASE_DELAY_MS,
        Settings.TOOL_RETRY_MAX_DELAY_MS,
        float(Settings.TOOL_RETRY_JITTER_RATIO),
    )


def _get_registry_runtime(registry: ToolRegistry) -> ToolRuntime:
    """按注册表和当前配置惰性创建、复用或替换 Runtime。"""
    config = _settings_runtime_config()
    with _runtime_lock:
        entry = _runtime_entries.get(registry)
        if (
            entry is not None
            and entry.config == config
            and not entry.runtime.closed
        ):
            return entry.runtime

        if entry is not None:
            entry.runtime.close()

        runtime = ToolRuntime(
            registry,
            max_workers=config[0],
            retry_base_delay_ms=config[1],
            retry_max_delay_ms=config[2],
            retry_jitter_ratio=config[3],
        )
        _runtime_entries[registry] = _RuntimeEntry(runtime, config)
        return runtime


def get_tool_runtime() -> ToolRuntime:
    """返回默认工具注册表对应的进程内共享 Runtime。"""
    from app.tools.registry import tool_registry

    return _get_registry_runtime(tool_registry)


def execute_registered_tool(
    *,
    registry: ToolRegistry,
    tool_name: str,
    tool_args: dict | None = None,
    context: ToolInvocationContext | None = None,
) -> dict:
    """通过指定注册表的共享 Runtime 执行工具。"""
    try:
        runtime = _get_registry_runtime(registry)
    except Exception as exc:
        spec = registry.get(tool_name)
        compatible_input = tool_args if isinstance(tool_args, dict) else {}
        return tool_error(
            tool_name,
            compatible_input,
            f"tool runtime is unavailable: {exc}",
            error_code=ToolErrorCode.RUNTIME_UNAVAILABLE.value,
            retryable=False,
            runtime={
                "tool_version": spec.version if spec is not None else "",
                "attempt_count": 0,
                "max_attempts": 0,
                "elapsed_ms": 0,
                "timeout": spec.timeout if spec is not None else None,
                "timed_out": False,
                "attempts": [],
            },
        )
    return runtime.execute(tool_name, tool_args, context=context)


def close_tool_runtime() -> None:
    """幂等关闭全部共享 Runtime，不等待正在运行的 Python 线程。"""
    with _runtime_lock:
        runtimes = [entry.runtime for entry in _runtime_entries.values()]
        _runtime_entries.clear()

    for runtime in runtimes:
        runtime.close()


def reset_tool_runtime() -> None:
    """关闭并清空共享 Runtime，供隔离测试或进程级重置使用。"""
    close_tool_runtime()


__all__ = [
    "ToolRuntime",
    "close_tool_runtime",
    "execute_registered_tool",
    "get_tool_runtime",
    "reset_tool_runtime",
]
