"""工具策略与参数指纹测试。"""

import pytest
from pydantic import BaseModel

from app.contracts.chat import ToolMode
from app.tools.context import PrincipalType, ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.policy import (
    ToolPolicyDecisionType,
    ToolPolicyEngine,
    build_tool_args_fingerprint,
)


class PolicyArgs(BaseModel):
    value: int


def _spec(**overrides) -> ToolSpec:
    values = {
        "name": "policy_tool",
        "version": "1.2.3",
        "description": "策略测试工具",
        "output_model": PolicyArgs,
        "handler": lambda args, context: {},
        "args_model": PolicyArgs,
        "required_permissions": ("document:write",),
    }
    values.update(overrides)
    return ToolSpec(**values)


def _context(**overrides) -> ToolInvocationContext:
    values = {
        "principal_id": "svc",
        "principal_type": PrincipalType.SERVICE,
        "permissions": ("document:write",),
        "source": "test",
    }
    values.update(overrides)
    return ToolInvocationContext(**values)


def test_fingerprint_is_stable_and_binds_identity_version_and_args():
    first = build_tool_args_fingerprint(
        tool_name="tool_a",
        tool_version="1.0.0",
        args={"a": 1, "b": 2},
    )
    reordered = build_tool_args_fingerprint(
        tool_name="tool_a",
        tool_version="1.0.0",
        args={"b": 2, "a": 1},
    )
    assert first == reordered
    assert first.startswith("sha256:")
    assert first != build_tool_args_fingerprint(
        tool_name="tool_a",
        tool_version="1.0.1",
        args={"a": 1, "b": 2},
    )
    assert first != build_tool_args_fingerprint(
        tool_name="tool_b",
        tool_version="1.0.0",
        args={"a": 1, "b": 2},
    )
    assert first != build_tool_args_fingerprint(
        tool_name="tool_a",
        tool_version="1.0.0",
        args={"a": 1, "b": 3},
    )


def test_policy_checks_permissions():
    spec = _spec()
    decision = ToolPolicyEngine().evaluate(
        spec=spec,
        args={"value": 1},
        context=_context(permissions=()),
        args_fingerprint="sha256:test",
    )
    assert decision.decision is ToolPolicyDecisionType.DENY
    assert decision.error_code.value == "permission_denied"
    assert decision.missing_permissions == ("document:write",)


@pytest.mark.parametrize("name", ["search_knowledge_base", "find_documents", "search_document", "get_chunk_context"])
def test_rag_only_preserves_knowledge_base_permission_gate(name):
    spec = _spec(name=name, required_permissions=("knowledge_base:read",))
    for permissions, expected in [
        ((), ToolPolicyDecisionType.DENY),
        (("knowledge_base:read",), ToolPolicyDecisionType.ALLOW),
    ]:
        decision = ToolPolicyEngine().evaluate(
            spec=spec, args={"value": 1}, args_fingerprint="sha256:test",
            context=_context(tool_mode=ToolMode.RAG_ONLY, permissions=permissions),
        )
        assert decision.decision is expected
        if not permissions:
            assert decision.error_code.value == "permission_denied"
