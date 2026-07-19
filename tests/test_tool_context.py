"""工具调用上下文规范化测试。"""

from dataclasses import fields

import pytest

from app.tools.context import PrincipalType, ToolInvocationContext


def test_context_normalizes_and_freezes_request_fields():
    context = ToolInvocationContext(
        trace_id=" trace-1 ",
        session_id=" session-1 ",
        principal_id=" service-1 ",
        principal_type="service",
        permissions=(
            " knowledge_base:read ",
            "knowledge_base:read",
            "ticket:create",
        ),
        source=" test ",
        idempotency_key=" idem-1 ",
    )

    assert context.trace_id == "trace-1"
    assert context.session_id == "session-1"
    assert context.principal_id == "service-1"
    assert context.source == "test"
    assert context.idempotency_key == "idem-1"
    assert context.principal_type is PrincipalType.SERVICE
    assert context.permissions == (
        "knowledge_base:read",
        "ticket:create",
    )
    assert {field.name for field in fields(ToolInvocationContext)} == {
        "trace_id",
        "session_id",
        "principal_id",
        "principal_type",
        "permissions",
        "source",
        "idempotency_key",
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("permissions", "knowledge_base:read"),
        ("permissions", ("knowledge_base:*",)),
    ],
)
def test_context_rejects_invalid_permissions(field_name, value):
    with pytest.raises(ValueError):
        ToolInvocationContext(**{field_name: value})
