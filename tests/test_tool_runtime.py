"""Tool Runtime V2 单元测试。"""

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.errors import ToolErrorCode, ToolExecutionError
from app.tools.registry import ToolRegistry
from app.tools.result import tool_error, tool_success
from app.tools.runtime import (
    ToolRuntime,
    get_tool_runtime,
    reset_tool_runtime,
)


class RuntimeArgs(BaseModel):
    """测试工具输入。"""

    value: int


class RuntimeOutput(BaseModel):
    """测试工具输出。"""

    value: int


def make_spec(
    handler: Callable[[dict, ToolInvocationContext], dict],
    **overrides: Any,
) -> ToolSpec:
    """构造独立的 Runtime 测试契约。"""
    values: dict[str, Any] = {
        "name": "runtime_echo",
        "version": "2.1.0",
        "description": "Runtime 测试工具",
        "args_schema": {"value": "integer"},
        "output_schema": {"value": "integer"},
        "handler": handler,
        "args_model": RuntimeArgs,
        "timeout": 1.0,
        "max_retries": 0,
        "idempotent": True,
    }
    values.update(overrides)
    return ToolSpec(**values)


@pytest.fixture(autouse=True)
def isolate_shared_runtime():
    """避免全局 Runtime 在线程和配置测试之间串用。"""
    reset_tool_runtime()
    yield
    reset_tool_runtime()


@pytest.fixture
def runtime_factory():
    """创建测试 Runtime，并在用例结束时幂等关闭。"""
    runtimes: list[ToolRuntime] = []

    def build(
        registry: ToolRegistry,
        *,
        sleep_func: Callable[[float], None] = time.sleep,
        random_func: Callable[[float, float], float] = (
            lambda _low, _high: 1.0
        ),
        retry_base_delay_ms: int = 0,
    ) -> ToolRuntime:
        runtime = ToolRuntime(
            registry,
            max_workers=2,
            retry_base_delay_ms=retry_base_delay_ms,
            retry_max_delay_ms=max(retry_base_delay_ms, 1000),
            retry_jitter_ratio=0.2,
            sleep_func=sleep_func,
            random_func=random_func,
        )
        runtimes.append(runtime)
        return runtime

    yield build

    for runtime in runtimes:
        runtime.close()


def test_runtime_executes_validated_args_once(runtime_factory):
    """正常执行只调用一次 handler，并覆盖不可信的工具身份字段。"""
    seen: list[tuple[dict, ToolInvocationContext]] = []

    def handler(args, context):
        seen.append((args, context))
        return tool_success("wrong_name", {"value": 999}, {"value": args["value"]})

    registry = ToolRegistry()
    registry.register(make_spec(handler))
    runtime = runtime_factory(registry)
    context = ToolInvocationContext(trace_id="runtime-trace")

    result = runtime.execute(
        "runtime_echo",
        {"value": "7"},
        context=context,
    )

    assert seen == [({"value": 7}, context)]
    assert result["ok"] is True
    assert result["tool_name"] == "runtime_echo"
    assert result["tool_input"] == {"value": 7}
    assert result["error"] == ""
    assert result["error_code"] == ""
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["max_attempts"] == 1
    assert result["runtime"]["timed_out"] is False
    assert result["runtime"]["elapsed_ms"] >= 0


def test_runtime_rejects_unknown_tool_without_attempt(runtime_factory):
    """未知工具不进入线程池。"""
    runtime = runtime_factory(ToolRegistry())

    result = runtime.execute("missing_tool", {"value": 1})

    assert result["error_code"] == "unknown_tool"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 0
    assert result["runtime"]["max_attempts"] == 0
    assert result["runtime"]["attempts"] == []


def test_runtime_rejects_invalid_args_before_handler(runtime_factory):
    """参数校验失败不执行 handler，也不重试。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": 1})

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    runtime = runtime_factory(registry)

    result = runtime.execute("runtime_echo", {"value": "not-an-int"})

    assert calls == 0
    assert result["error_code"] == "invalid_arguments"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 0


def test_timeout_returns_early_and_never_retries(runtime_factory):
    """超时只停止等待，后台 handler 自行结束且 Runtime 不立即重试。"""
    calls = 0
    completed = threading.Event()

    def handler(args, context):
        nonlocal calls
        calls += 1
        time.sleep(0.12)
        completed.set()
        return tool_success("runtime_echo", args, {"value": args["value"]})

    registry = ToolRegistry()
    registry.register(make_spec(handler, timeout=0.02, max_retries=2))
    runtime = runtime_factory(registry)

    started_at = time.perf_counter()
    result = runtime.execute("runtime_echo", {"value": 1})
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.1
    assert result["error_code"] == "timeout"
    assert result["retryable"] is True
    assert result["runtime"]["timed_out"] is True
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["max_attempts"] == 3
    assert completed.wait(timeout=1.0)
    assert calls == 1


def test_retryable_error_succeeds_with_exponential_backoff(runtime_factory):
    """幂等工具对显式可重试错误退避后再次执行。"""
    calls = 0
    sleeps: list[float] = []

    def handler(args, context):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ToolExecutionError(
                "dependency unavailable",
                code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
                retryable=True,
            )
        return tool_success("runtime_echo", args, {"value": args["value"]})

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    runtime = runtime_factory(
        registry,
        sleep_func=sleeps.append,
        retry_base_delay_ms=100,
    )

    result = runtime.execute("runtime_echo", {"value": 3})

    assert result["ok"] is True
    assert calls == 3
    assert sleeps == pytest.approx([0.1, 0.2])
    assert result["runtime"]["attempt_count"] == 3
    assert [item["error_code"] for item in result["runtime"]["attempts"]] == [
        "dependency_unavailable",
        "dependency_unavailable",
        "",
    ]


def test_retryable_error_exhaustion_keeps_retryable_flag(runtime_factory):
    """重试耗尽后保留错误本身的可重试性质。"""
    calls = 0
    sleeps: list[float] = []

    def handler(args, context):
        nonlocal calls
        calls += 1
        raise ToolExecutionError(
            "temporarily unavailable",
            code=ToolErrorCode.DEPENDENCY_UNAVAILABLE,
            retryable=True,
        )

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    runtime = runtime_factory(
        registry,
        sleep_func=sleeps.append,
        retry_base_delay_ms=10,
    )

    result = runtime.execute("runtime_echo", {"value": 3})

    assert calls == 3
    assert len(sleeps) == 2
    assert result["ok"] is False
    assert result["error_code"] == "dependency_unavailable"
    assert result["retryable"] is True
    assert result["runtime"]["attempt_count"] == 3


def test_unknown_exception_is_not_retried(runtime_factory):
    """普通异常默认不可重试。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        raise ValueError("handler bug")

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    result = runtime_factory(registry).execute("runtime_echo", {"value": 3})

    assert calls == 1
    assert result["error_code"] == "execution_error"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 1


def test_handler_timeout_error_is_an_unknown_exception(runtime_factory):
    """handler 主动抛普通 TimeoutError 不等同于 Runtime 等待超时。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        raise TimeoutError("handler timeout exception")

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    result = runtime_factory(registry).execute("runtime_echo", {"value": 3})

    assert calls == 1
    assert result["error_code"] == "execution_error"
    assert result["retryable"] is False
    assert result["runtime"]["timed_out"] is False
    assert result["runtime"]["attempt_count"] == 1


def test_non_idempotent_tool_is_defensively_limited_to_one_attempt(
    runtime_factory,
):
    """即使绕过契约校验，非幂等工具仍不会执行第二次。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        raise ToolExecutionError(
            "rate limited",
            code=ToolErrorCode.RATE_LIMITED,
            retryable=True,
        )

    spec = make_spec(handler)
    object.__setattr__(spec, "idempotent", False)
    object.__setattr__(spec, "max_retries", 3)
    registry = ToolRegistry()
    registry.register(spec)

    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert calls == 1
    assert result["error_code"] == "rate_limited"
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["max_attempts"] == 1


@pytest.mark.parametrize(
    "returned_value",
    [
        None,
        "success",
        [],
        123,
        {"status": "error", "ok": True},
        {"status": "success", "ok": False},
    ],
)
def test_runtime_rejects_invalid_handler_results(
    runtime_factory,
    returned_value,
):
    """非字典或状态矛盾的 handler 返回值统一归类为非法结果。"""
    registry = ToolRegistry()
    registry.register(make_spec(lambda args, context: returned_value))

    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert result["error_code"] == "invalid_result"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 1


def test_handler_error_dict_is_conservative_and_not_retried(runtime_factory):
    """普通失败字典不会被推断为可重试错误。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_error("runtime_echo", args, "business failure")

    registry = ToolRegistry()
    registry.register(make_spec(handler, max_retries=2))
    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert calls == 1
    assert result["error_code"] == "execution_error"
    assert result["retryable"] is False


def test_output_model_validates_and_normalizes_success(runtime_factory):
    """成功输出通过 output_model 后使用模型规范化值。"""
    def handler(args, context):
        return tool_success("runtime_echo", args, {"value": "9"})

    registry = ToolRegistry()
    registry.register(make_spec(handler, output_model=RuntimeOutput))

    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert result["ok"] is True
    assert result["output"] == {"value": 9}


def test_output_model_failure_is_not_retried(runtime_factory):
    """输出模型校验失败立即返回，不重复调用 handler。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": "invalid"})

    registry = ToolRegistry()
    registry.register(
        make_spec(handler, output_model=RuntimeOutput, max_retries=2)
    )

    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert calls == 1
    assert result["error_code"] == "output_validation_error"
    assert result["retryable"] is False
    assert result["runtime"]["attempt_count"] == 1


def test_runtime_metadata_contains_stable_fields(runtime_factory):
    """每个已注册工具结果包含稳定元数据。"""
    registry = ToolRegistry()
    registry.register(
        make_spec(
            lambda args, context: tool_success(
                "runtime_echo",
                args,
                {"value": args["value"]},
            )
        )
    )

    result = runtime_factory(registry).execute("runtime_echo", {"value": 1})

    assert {
        "tool_version",
        "attempt_count",
        "max_attempts",
        "elapsed_ms",
        "timeout",
        "timed_out",
    } <= result["runtime"].keys()
    assert result["runtime"]["tool_version"] == "2.1.0"
    assert result["runtime"]["timeout"] == 1.0


def test_runtime_close_is_idempotent_and_disables_execute(runtime_factory):
    """关闭操作可重复，关闭后的实例返回 runtime_unavailable。"""
    calls = 0

    def handler(args, context):
        nonlocal calls
        calls += 1
        return tool_success("runtime_echo", args, {"value": 1})

    registry = ToolRegistry()
    registry.register(make_spec(handler))
    runtime = runtime_factory(registry)

    runtime.close()
    runtime.close()
    result = runtime.execute("runtime_echo", {"value": 1})

    assert calls == 0
    assert result["error_code"] == "runtime_unavailable"
    assert result["runtime"]["attempt_count"] == 0


def test_legacy_result_helpers_keep_positional_calls():
    """结果辅助函数的三个旧位置参数继续有效。"""
    success = tool_success("echo", {"value": 1}, {"value": 1})
    error = tool_error("echo", {"value": 1}, "failed")

    assert success["ok"] is True
    assert success["error_code"] == ""
    assert success["runtime"] == {}
    assert error["ok"] is False
    assert error["error_code"] == "execution_error"
    assert error["retryable"] is False
    assert error["runtime"] == {}


def test_registry_dispatch_uses_runtime_protocol():
    """Registry 兼容入口不会绕过 Runtime。"""
    registry = ToolRegistry()
    registry.register(
        make_spec(
            lambda args, context: tool_success(
                "runtime_echo",
                args,
                {"value": args["value"]},
            )
        )
    )

    result = registry.dispatch("runtime_echo", {"value": "4"})

    assert result["tool_input"] == {"value": 4}
    assert result["runtime"]["attempt_count"] == 1


def test_registry_dispatch_returns_runtime_extensions(tool_registry):
    """直接测试通过 Registry 分发并获得 Runtime 扩展字段。"""
    result = tool_registry.dispatch(
        "simple_calculator",
        {"expression": "1 + 2"},
        context=ToolInvocationContext(trace_id="runtime-trace"),
    )

    assert result["ok"] is True
    assert result["output"]["result"] == 3
    assert result["runtime"]["attempt_count"] == 1
    assert result["runtime"]["tool_version"] == "1.0.0"


def test_shared_runtime_replaces_instance_when_config_changes(
    settings_env,
):
    """配置变化时不会静默复用参数已经过期的 Runtime。"""
    first = get_tool_runtime()
    settings_env(tool_runtime_max_workers=9)

    second = get_tool_runtime()

    assert second is not first
    assert first.closed is True
