"""不可变本地工具集合、配置缓存和重复名称回归。"""

import pytest

from app.agent import framework_tools
from app.agent import runtime as agent_runtime
from app.agent.runtime import get_compiled_agent_runtime
from app.tools import catalog
from app.tools.context import ToolInvocationContext
from tests.support.local_tools import visible_tools


def test_catalog_caches_configuration_and_preserves_order(settings_factory):
    settings = settings_factory()
    changed = settings_factory(agent_max_total_cost_ms=60_000)
    specs = catalog.get_local_tool_specs(settings)
    tools = framework_tools.get_local_tools(settings)
    assert isinstance(specs, tuple) and isinstance(tools, tuple)
    assert specs is catalog.get_local_tool_specs(settings)
    assert tools is framework_tools.get_local_tools(settings)
    expected = (
        "search_knowledge_base", "web_search", "web_fetch", "web_pdf_fetch",
        "compare_documents", "list_documents", "get_chunk_text",
    )
    assert catalog.get_local_tool_names(settings) == expected
    assert tuple(tool.name for tool in tools) == expected
    assert catalog.get_local_tool_names(changed) == expected
    assert catalog.get_local_tool_specs(changed)[0].timeout == 60.0
    assert catalog.get_local_tool_specs(changed) is not specs
    assert framework_tools.get_local_tools(settings) is tools
    assert all(not hasattr(tool, "project_context") for tool in tools)
    with pytest.raises(TypeError):
        tools[0] = tools[-1]


def test_catalog_rejects_duplicate_names_once(monkeypatch, settings_factory):
    settings = settings_factory()
    duplicate = catalog.get_local_tool_specs(settings)[0]
    catalog.get_local_tool_specs.cache_clear()
    monkeypatch.setattr(catalog, "ToolSpec", lambda **kwargs: duplicate)
    with pytest.raises(ValueError, match="重复"):
        catalog.get_local_tool_specs(settings)


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
    settings = settings_factory()
    contexts = [ToolInvocationContext(), ToolInvocationContext(principal_id="other", permissions=("web:search",)), ToolInvocationContext(permissions=("knowledge_base:read",))]
    tools = framework_tools.get_local_tools(settings)
    agents = [get_compiled_agent_runtime(settings, tool_context=context) for context in contexts]
    assert all(agent is agents[0] for agent in agents)
    assert agents[0] is get_compiled_agent_runtime(settings)
    names = [[tool.name for tool in visible_tools(tools, context=context)] for context in contexts]
    assert names[0] == []
    assert names[1] == ["web_search"]
    assert names[2] == ["search_knowledge_base", "compare_documents", "list_documents", "get_chunk_text"]


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
