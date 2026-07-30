"""Tool Contract V2 契约校验测试。"""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from app.tools.context import ToolInvocationContext
from app.tools.contracts import (
    ToolRiskLevel,
    ToolRiskLevel as ContractToolRiskLevel,
    ToolSpec,
)
from app.tools.contracts import ToolSpec as ContractToolSpec
from app.tools.registry import (
    ToolRegistry,
    ToolRiskLevel as RegistryToolRiskLevel,
)
from app.tools.registry import ToolSpec as RegistryToolSpec


class EchoArgs(BaseModel):
    """回显工具参数。"""

    value: str


class EchoOutput(BaseModel):
    """回显工具输出。"""

    value: str


def echo_handler(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """返回输入值，供契约测试使用。"""
    return {
        "status": "success",
        "ok": True,
        "tool_name": "echo_tool",
        "tool_input": args,
        "output": {"value": args["value"]},
        "error": "",
    }


def _make_spec(**overrides: Any) -> ToolSpec:
    """构造可按字段覆盖的合法测试契约。"""
    values = {
        "name": "echo_tool",
        "version": "1.0.0",
        "description": "回显输入",
        "args_schema": {"value": "string"},
        "output_schema": {"value": "string"},
        "handler": echo_handler,
        "args_model": EchoArgs,
    }
    values.update(overrides)
    return ToolSpec(**values)


def test_tool_contract_public_exports_are_compatible():
    """公共入口、契约模块和旧注册表入口导出同一类型。"""
    assert ToolSpec is ContractToolSpec is RegistryToolSpec
    assert (
        ToolRiskLevel
        is ContractToolRiskLevel
        is RegistryToolRiskLevel
    )


def test_tool_spec_normalizes_and_serializes_contract():
    """合法契约会被规范化，并可安全序列化给治理与检查端。"""
    spec = _make_spec(
        name="  echo_tool  ",
        version=" 1.0.0-beta.1 ",
        description="  回显输入  ",
        output_model=EchoOutput,
        risk_level="internal_write",
        required_permissions=(
            " knowledge_base:read ",
            "ticket:create",
            "knowledge_base:read",
        ),
        timeout=5,
    )

    assert spec.name == "echo_tool"
    assert spec.version == "1.0.0-beta.1"
    assert spec.description == "回显输入"
    assert spec.risk_level is ToolRiskLevel.INTERNAL_WRITE
    assert spec.timeout == 5.0
    assert isinstance(spec.timeout, float)
    assert spec.required_permissions == (
        "knowledge_base:read",
        "ticket:create",
    )

    contract_spec = spec.to_contract_dict()
    json.dumps(contract_spec)

    assert "handler" not in contract_spec
    assert contract_spec["args_model"] == "EchoArgs"
    assert contract_spec["output_model"] == "EchoOutput"
    assert contract_spec["required_permissions"] == [
        "knowledge_base:read",
        "ticket:create",
    ]


@pytest.mark.parametrize(
    "name",
    ["", "   ", "Echo Tool", "echo-tool", "2echo", "搜索工具"],
)
def test_tool_spec_rejects_invalid_name(name):
    """工具名必须使用受限的小写 snake_case 格式。"""
    with pytest.raises(ValueError, match="字段 'name' 非法"):
        _make_spec(name=name)


@pytest.mark.parametrize("description", ["", "   "])
def test_tool_spec_rejects_empty_description(description):
    """工具描述去除首尾空格后不能为空。"""
    with pytest.raises(ValueError, match="字段 'description' 非法"):
        _make_spec(description=description)


@pytest.mark.parametrize("version", ["", "v1", "latest", "version-2"])
def test_tool_spec_rejects_invalid_version(version):
    """工具版本必须遵循基础 SemVer。"""
    with pytest.raises(ValueError, match="字段 'version' 非法"):
        _make_spec(version=version)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("inf"), float("nan")],
)
def test_tool_spec_rejects_invalid_timeout(timeout):
    """超时声明必须是有限正数。"""
    with pytest.raises(ValueError, match="字段 'timeout' 非法"):
        _make_spec(timeout=timeout)


@pytest.mark.parametrize("max_retries", [-1, True, False])
def test_tool_spec_rejects_invalid_max_retries(max_retries):
    """重试次数必须是非负整数，并显式拒绝 bool。"""
    with pytest.raises(ValueError, match="字段 'max_retries' 非法"):
        _make_spec(max_retries=max_retries)


@pytest.mark.parametrize(
    "required_permissions",
    [
        "knowledge_base:read",
        ("",),
        ("   ",),
        ("Knowledge Base:Read",),
        ("read",),
    ],
)
def test_tool_spec_rejects_invalid_permissions(required_permissions):
    """权限必须以规范的字符串序列声明。"""
    with pytest.raises(
        ValueError,
        match="字段 'required_permissions' 非法",
    ):
        _make_spec(required_permissions=required_permissions)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("args_schema", []),
        ("output_schema", []),
        ("handler", None),
        ("args_model", object),
        ("output_model", object),
        ("idempotent", 1),
        ("requires_approval", 0),
        ("risk_level", "unknown"),
    ],
)
def test_tool_spec_rejects_invalid_base_field_types(
    field_name,
    invalid_value,
):
    """基础契约字段在创建阶段执行明确的类型校验。"""
    with pytest.raises(ValueError) as exc_info:
        _make_spec(**{field_name: invalid_value})

    error = str(exc_info.value)
    assert "echo_tool" in error
    assert field_name in error
    assert "非法" in error


def test_non_idempotent_tool_cannot_declare_retries():
    """非幂等工具不能声明自动重试。"""
    with pytest.raises(ValueError, match="非幂等工具不能声明自动重试"):
        _make_spec(idempotent=False, max_retries=1)


@pytest.mark.parametrize(
    "risk_level",
    [ToolRiskLevel.EXTERNAL_WRITE, ToolRiskLevel.DESTRUCTIVE],
)
def test_high_risk_tool_requires_approval(risk_level):
    """外部写和破坏性工具必须要求人工审批。"""
    with pytest.raises(ValueError, match="必须要求人工审批"):
        _make_spec(
            risk_level=risk_level,
            requires_approval=False,
        )


def test_registry_rejects_duplicate_tool_name():
    """注册表拒绝第二个同名工具，避免静默覆盖。"""
    registry = ToolRegistry()
    first_spec = _make_spec()
    registry.register(first_spec)

    with pytest.raises(ValueError, match="已注册，禁止静默覆盖"):
        registry.register(_make_spec(description="另一个回显工具"))

    assert registry.get("echo_tool") is first_spec


def test_registry_rejects_non_tool_spec():
    """注册表只接受 ToolSpec 实例。"""
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="spec 必须是 ToolSpec"):
        registry.register(object())
