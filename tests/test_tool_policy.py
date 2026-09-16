"""工具策略与参数指纹测试。"""

from pydantic import BaseModel

from app.contracts.chat import ToolMode
from app.tools.context import PrincipalType, ToolInvocationContext
from app.tools.contracts import ToolRiskLevel, ToolSpec
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
        "idempotency_key": "idem-1",
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


def test_policy_checks_permissions_before_idempotency_and_approval():
    spec = _spec(
        risk_level=ToolRiskLevel.EXTERNAL_WRITE,
        requires_approval=True,
    )
    decision = ToolPolicyEngine().evaluate(
        spec=spec,
        args={"value": 1},
        context=_context(permissions=(), idempotency_key=""),
        args_fingerprint="sha256:test",
    )
    assert decision.decision is ToolPolicyDecisionType.DENY
    assert decision.error_code.value == "permission_denied"
    assert decision.missing_permissions == ("document:write",)


def test_policy_requires_idempotency_then_approval_for_external_write():
    spec = _spec(
        risk_level=ToolRiskLevel.EXTERNAL_WRITE,
        requires_approval=True,
    )
    missing_key = ToolPolicyEngine().evaluate(
        spec=spec,
        args={"value": 1},
        context=_context(idempotency_key=""),
        args_fingerprint="sha256:test",
    )
    assert missing_key.error_code.value == "missing_idempotency_key"

    approval = ToolPolicyEngine().evaluate(
        spec=spec,
        args={"value": 1},
        context=_context(),
        args_fingerprint="sha256:test",
    )
    assert approval.decision is ToolPolicyDecisionType.REQUIRE_APPROVAL
    assert approval.approval_required is True


def test_policy_rejects_anonymous_write_context():
    decision = ToolPolicyEngine().evaluate(
        spec=_spec(risk_level=ToolRiskLevel.INTERNAL_WRITE),
        args={"value": 1},
        context=_context(principal_type=PrincipalType.ANONYMOUS),
        args_fingerprint="sha256:test",
    )
    assert decision.error_code.value == "invalid_invocation_context"


def test_rag_only_preserves_knowledge_base_permission_gate():
    spec = _spec(name="search_knowledge_base", required_permissions=("knowledge_base:read",))
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
