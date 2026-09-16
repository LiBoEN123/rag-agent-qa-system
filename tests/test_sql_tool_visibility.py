"""SQL 配置开关与 database:read 请求权限的四象限可见性测试。"""

import pytest

from app.tools.context import ToolInvocationContext
from tests.support.local_tools import local_tools, visible_tools
from app.tools.catalog import get_local_tool_specs


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
    registry = get_local_tool_specs(settings)
    visible = {
        spec.name
        for spec in visible_tools(local_tools(registry),
            context=ToolInvocationContext(permissions=permissions)
        )
    }

    assert (SQL_TOOL_NAMES <= set([item.name for item in registry])) is enabled
    assert (SQL_TOOL_NAMES <= visible) is expected_visible
    if not enabled:
        assert len([item.name for item in registry]) == 8


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
