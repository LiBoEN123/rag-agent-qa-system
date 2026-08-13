"""SQL 配置开关与 database:read 请求权限的四象限可见性测试。"""

import pytest

from app.tools.context import ToolInvocationContext
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry, register_default_tools


SQL_TOOL_NAMES = {
    "list_sql_tables",
    "describe_sql_table",
    "query_sql_readonly",
}


@pytest.mark.parametrize(
    ("enabled", "permissions", "expected_visible"),
    [
        (False, (), False),
        (False, ("database:read",), False),
        (True, (), False),
        (True, ("database:read",), True),
    ],
)
def test_sql_tool_visibility_requires_configuration_and_permission(
    settings_factory,
    sql_business_db,
    enabled,
    permissions,
    expected_visible,
):
    overrides = {"sql_tool_enabled": enabled}
    if enabled:
        overrides.update(
            {
                "sql_database_url": f"sqlite:///{sql_business_db}",
                "sql_allowed_tables": ("students",),
            }
        )
    settings = settings_factory(**overrides)
    registry = ToolRegistry()
    register_default_tools(registry, settings=settings)
    visible = {
        spec.name
        for spec in LocalToolProvider(registry).list_specs(
            context=ToolInvocationContext(permissions=permissions)
        )
    }

    assert (SQL_TOOL_NAMES <= set(registry.names())) is enabled
    assert (SQL_TOOL_NAMES <= visible) is expected_visible
    if not enabled:
        assert len(registry.names()) == 8


def test_enabling_sql_does_not_grant_database_permission(
    settings_factory,
    sql_business_db,
):
    settings = settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students",),
    )

    assert settings.tool_default_permissions == ("knowledge_base:read",)
