"""工具调用上下文规范化测试。"""

import pytest

from app.tools.context import PrincipalType, ToolInvocationContext


def test_context_normalizes_and_freezes_request_fields():
    metadata = {"channel": " internal "}
    context = ToolInvocationContext(
        trace_id=" trace-1 ",
        request_id=" request-1 ",
        session_id=" session-1 ",
        principal_id=" service-1 ",
        principal_type="service",
        roles=(" reader ", "reader", "operator"),
        permissions=(
            " knowledge_base:read ",
            "knowledge_base:read",
            "ticket:create",
        ),
        source=" test ",
        metadata=metadata,
    )
    metadata["channel"] = "changed"

    assert context.trace_id == "trace-1"
    assert context.request_id == "request-1"
    assert context.session_id == "session-1"
    assert context.principal_id == "service-1"
    assert context.source == "test"
    assert context.principal_type is PrincipalType.SERVICE
    assert context.roles == ("reader", "operator")
    assert context.permissions == (
        "knowledge_base:read",
        "ticket:create",
    )
    assert context.metadata["channel"] == " internal "
    with pytest.raises(TypeError):
        context.metadata["new"] = "value"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("roles", "reader"),
        ("roles", ("Reader",)),
        ("permissions", "knowledge_base:read"),
        ("permissions", ("knowledge_base:*",)),
    ],
)
def test_context_rejects_invalid_roles_and_permissions(field_name, value):
    with pytest.raises(ValueError):
        ToolInvocationContext(**{field_name: value})


def test_context_rejects_sensitive_metadata_recursively():
    with pytest.raises(ValueError, match="敏感键"):
        ToolInvocationContext(metadata={"nested": {"access-token": "abc"}})

