"""不可变本地工具集合、配置缓存和重复名称回归。"""

import pytest

from app.agent import framework_tools
from app.agent import runtime as agent_runtime
from app.agent.runtime import get_compiled_agent_runtime
from app.tools import catalog
from app.tools.context import ToolInvocationContext
from tests.support.local_tools import visible_tools


def test_catalog_caches_configuration_and_preserves_sql_order(settings_factory):
    disabled = settings_factory(sql_tool_enabled=False)
    enabled = disabled.model_copy(update={"sql_tool_enabled": True})
    specs = catalog.get_local_tool_specs(disabled)
    tools = framework_tools.get_local_tools(disabled)
    assert isinstance(specs, tuple) and isinstance(tools, tuple)
    assert specs is catalog.get_local_tool_specs(disabled)
    assert tools is framework_tools.get_local_tools(disabled)
    assert [t.name for t in tools] == [s.name for s in specs]
    assert catalog.get_local_tool_names(enabled) == (*catalog.get_local_tool_names(disabled), "list_sql_tables", "describe_sql_table", "query_sql_readonly")
    assert len(framework_tools.get_local_tools(enabled)) == 11
    assert framework_tools.get_local_tools(disabled) is tools
    assert all(not hasattr(tool, "project_context") for tool in tools)
    with pytest.raises(TypeError):
        tools[0] = tools[-1]


def test_catalog_rejects_duplicate_names_once(monkeypatch, settings_factory):
    disabled = settings_factory(sql_tool_enabled=False)
    duplicate = catalog.get_local_tool_specs(disabled)[0]
    monkeypatch.setattr(catalog, "build_sql_readonly_tool_specs", lambda settings: (duplicate,))
    with pytest.raises(ValueError, match="重复"):
        catalog.get_local_tool_specs(disabled.model_copy(update={"sql_tool_enabled": True}))


def test_default_catalog_contract_is_fresh_and_complete(settings_factory):
    settings = settings_factory()
    from copy import deepcopy

    contracts = framework_tools.get_local_tool_contracts(settings)
    expected = deepcopy(contracts)
    specs = catalog.get_local_tool_specs(settings)
    assert [contract["name"] for contract in contracts] == [spec.name for spec in specs]
    for spec, contract in zip(specs, contracts, strict=True):
        assert contract["args_schema"] == spec.args_model.model_json_schema()
        assert contract["output_schema"] == spec.output_model.model_json_schema()
    contracts[0]["args_schema"]["properties"].clear()
    assert framework_tools.get_local_tool_contracts(settings) == expected


def test_cached_agent_and_tools_do_not_capture_principal_or_permissions(settings_factory):
    settings = settings_factory(sql_tool_enabled=False)
    contexts = [ToolInvocationContext(), ToolInvocationContext(principal_id="other", permissions=("web:search",)), ToolInvocationContext(permissions=("knowledge_base:read",))]
    tools = framework_tools.get_local_tools(settings)
    agents = [get_compiled_agent_runtime(settings, tool_context=context) for context in contexts]
    assert all(agent is agents[0] for agent in agents)
    assert agents[0] is get_compiled_agent_runtime(settings)
    names = [[tool.name for tool in visible_tools(tools, context=context)] for context in contexts]
    assert names[0] == ["simple_calculator"]
    assert names[1] == ["web_search", "simple_calculator"]
    assert names[2] == ["search_knowledge_base", "compare_documents", "simple_calculator", "list_documents", "get_chunk_text"]


def test_resolve_agent_tools_passes_same_explicit_settings(monkeypatch, settings_factory):
    explicit_settings = settings_factory()
    context = ToolInvocationContext()
    local_settings = []
    mcp_settings = []

    def get_local_tools(settings):
        local_settings.append(settings)
        return ()

    def list_mcp_tools(received_context, *, settings):
        assert received_context is context
        mcp_settings.append(settings)
        return []

    monkeypatch.setattr(framework_tools, "get_local_tools", get_local_tools)
    monkeypatch.setattr(framework_tools, "list_mcp_tools_for_context", list_mcp_tools)

    assert framework_tools.resolve_agent_tools(context, explicit_settings) == []
    assert len(local_settings) == len(mcp_settings) == 1
    assert local_settings[0] is mcp_settings[0] is explicit_settings


def test_compiled_agent_runtime_passes_explicit_mcp_settings(monkeypatch, settings_factory):
    explicit_settings = settings_factory()
    context = ToolInvocationContext()
    mcp_settings = []
    cached_agent = object()

    def list_mcp_tools(received_context, *, settings):
        assert received_context is context
        mcp_settings.append(settings)
        return []

    monkeypatch.setattr(agent_runtime, "list_mcp_tools_for_context", list_mcp_tools)
    monkeypatch.setattr(agent_runtime, "build_framework_agent", lambda settings, **kwargs: cached_agent)

    assert get_compiled_agent_runtime(explicit_settings, tool_context=context) is cached_agent
    assert len(mcp_settings) == 1
    assert mcp_settings[0] is explicit_settings
