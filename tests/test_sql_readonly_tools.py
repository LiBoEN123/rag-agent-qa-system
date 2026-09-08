"""三个 SQL ToolSpec、Runtime 执行与 Framework Agent 集成测试。"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from app.agent import framework_agent, framework_tools
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolAuditOutputMode, ToolRiskLevel
from tests.support.local_tools import local_tools
from app.tools.catalog import get_local_tool_specs
from tests.support.local_tools import invoke_local_tool
from tests.support.agent_runtime import build_agent_harness


SQL_TOOL_NAMES = {
    "list_sql_tables",
    "describe_sql_table",
    "query_sql_readonly",
}


@pytest.fixture
def sql_settings(settings_factory, sql_business_db):
    return settings_factory(
        sql_tool_enabled=True,
        sql_database_url=f"sqlite:///{sql_business_db}",
        sql_allowed_tables=("students", "courses"),
        tool_audit_enabled=False,
    )


@pytest.fixture
def sql_registry(sql_settings):
    registry = get_local_tool_specs(sql_settings)
    return registry


def test_sql_tool_specs_use_existing_readonly_governance(sql_registry, sql_settings):
    for name in SQL_TOOL_NAMES:
        spec = next((item for item in sql_registry if item.name == name), None)
        assert spec is not None
        assert spec.risk_level is ToolRiskLevel.READ_ONLY
        assert spec.required_permissions == ("database:read",)
        assert spec.idempotent is True
        assert spec.requires_approval is False
        assert spec.max_retries == 0
        assert spec.timeout == sql_settings.sql_tool_timeout_seconds
        assert spec.audit_output_mode is ToolAuditOutputMode.SUMMARY
        assert spec.args_model.model_config.get("extra") == "forbid"
        assert spec.output_model is not None


def test_query_args_and_outputs_are_pydantic_validated(sql_registry):
    query_spec = next((item for item in sql_registry if item.name == "query_sql_readonly"), None)

    with pytest.raises(ValidationError):
        query_spec.args_model.model_validate(
            {"sql": "SELECT * FROM students", "unexpected": True}
        )
    validated = query_spec.output_model.model_validate(
        {
            "columns": ["id"],
            "rows": [[1]],
            "row_count": 1,
            "truncated": False,
            "dialect": "sqlite",
            "execution_ms": 0,
            "content_is_untrusted": True,
        }
    )
    assert validated.rows == [[1]]


def test_runtime_permission_gate_runs_before_sql_handler(sql_registry):
    denied = invoke_local_tool(
        registry=sql_registry,
        tool_name="list_sql_tables",
        tool_args={},
        context=ToolInvocationContext(source="sql_tool_test"),
    )

    assert denied["ok"] is False
    assert denied["error_code"] == "sql_permission_denied"
    assert denied["governance"]["missing_permissions"] == ["database:read"]


def test_runtime_executes_three_sql_tools_with_structured_outputs(sql_registry):
    context = ToolInvocationContext(
        permissions=("database:read",),
        source="sql_tool_test",
    )
    calls = [
        ("list_sql_tables", {}),
        ("describe_sql_table", {"table_name": "students"}),
        (
            "query_sql_readonly",
            {"sql": "SELECT id, name FROM students ORDER BY id", "max_rows": 2},
        ),
    ]

    results = [
        invoke_local_tool(
            registry=sql_registry,
            tool_name=name,
            tool_args=args,
            context=context,
        )
        for name, args in calls
    ]

    assert all(result["ok"] is True for result in results)
    assert all(result["output"]["content_is_untrusted"] is True for result in results)
    assert results[0]["output"]["tables"] == ["students", "courses"]
    assert results[1]["output"]["column_count"] == 4
    assert results[2]["output"]["rows"] == [[1, "张三"], [2, "李四"]]


def test_runtime_returns_safe_specific_sql_error(sql_registry):
    result = invoke_local_tool(
        registry=sql_registry,
        tool_name="query_sql_readonly",
        tool_args={"sql": "DELETE FROM students"},
        context=ToolInvocationContext(
            permissions=("database:read",),
            source="sql_tool_test",
        ),
    )

    assert result["ok"] is False
    assert result["error_code"] == "sql_write_forbidden"
    assert result["retryable"] is False
    assert "只允许" in result["error"]


class SqlToolCallingModel:
    """一次请求三个 SQL 工具，再返回最终回答的离线假模型。"""

    def __init__(self):
        self.bound_tool_names: list[str] = []
        self.received_messages: list[list] = []
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_sql_tables",
                        "args": {},
                        "id": "sql-list",
                        "type": "tool_call",
                    },
                    {
                        "name": "describe_sql_table",
                        "args": {"table_name": "students"},
                        "id": "sql-describe",
                        "type": "tool_call",
                    },
                    {
                        "name": "query_sql_readonly",
                        "args": {
                            "sql": "SELECT id, name FROM students ORDER BY id LIMIT 1"
                        },
                        "id": "sql-query",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="数据库只读工具调用完成"),
        ]

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        return self.responses.pop(0)


def test_framework_agent_calls_all_sql_tools_through_registry_runtime(
    monkeypatch,
    sql_registry,
    sql_settings,
):
    model = SqlToolCallingModel()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_local_tools",
        lambda settings: local_tools(sql_registry),
    )
    context = ToolInvocationContext(
        trace_id="trace-sql-agent",
        permissions=("knowledge_base:read", "database:read"),
        source="sql_agent_test",
    )

    output = build_agent_harness(sql_settings).invoke(
        {
            "original_question": "查看学生表",
            "effective_question": "查看学生表",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent", "router_llm_used": False},
            "trace_id": context.trace_id,
            "session_id": context.session_id,
        },
        context=context,
    )

    assert SQL_TOOL_NAMES <= set(model.bound_tool_names)
    tool_messages = [
        message
        for message in model.received_messages[1]
        if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 3
    assert all(json.loads(message.content)["ok"] for message in tool_messages)
    assert output["result"]["answer"] == "数据库只读工具调用完成"
