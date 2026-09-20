"""不可变本地工具集合、配置缓存和重复名称回归。"""

import pytest

from app.agent import framework_tools
from app.agent import runtime as agent_runtime
from app.agent.runtime import get_compiled_agent_runtime
from app.tools import catalog
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolAuditOutputMode, ToolRiskLevel
from app.tools.schemas import (
    FindDocumentsArgs, FindDocumentsOutput, GetChunkContextArgs, GetChunkContextOutput,
    SearchDocumentArgs, SearchDocumentOutput,
)
from tests.support.local_tools import visible_tools


def test_knowledge_tool_description_explains_candidate_evidence(settings_factory):
    spec = catalog.get_local_tool_specs(settings_factory())[0]
    assert spec.name == "search_knowledge_base"
    for guidance in ("候选 evidence", "不生成最终答案", "阅读结果", "修改 query", "不可信数据"):
        assert guidance in spec.description


@pytest.mark.parametrize("name,args_model,output_model", [
    ("find_documents", FindDocumentsArgs, FindDocumentsOutput),
    ("search_document", SearchDocumentArgs, SearchDocumentOutput),
])
def test_document_tools_publish_governed_contract(settings_factory, name, args_model, output_model):
    specs = catalog.get_local_tool_specs(settings_factory())
    spec = next(spec for spec in specs if spec.name == name)
    assert spec.args_model is args_model
    assert spec.output_model is output_model
    assert spec.risk_level is ToolRiskLevel.READ_ONLY
    assert spec.required_permissions == ("knowledge_base:read",)
    assert spec.audit_output_mode is ToolAuditOutputMode.SUMMARY
    assert spec.idempotent is True and spec.requires_approval is False
    assert spec.max_retries == 0
    assert spec.timeout == (5.0 if name == "find_documents" else specs[0].timeout)


def test_chunk_lookup_publishes_integer_locator_contract(settings_factory):
    tool = next(tool for tool in framework_tools.get_local_tools(settings_factory())
                if tool.name == "get_chunk_text")
    assert tool.project_spec.version == tool.metadata["tool_version"] == "1.1.0"
    locator = tool.tool_call_schema.model_json_schema()["properties"]["chunk_id"]
    assert locator["type"] == "integer"
    assert locator["minimum"] == 1
    assert "anyOf" not in locator
    for guidance in ("Document.metadata.chunk_id", "不能使用 Document.id", "page_content", "完整"):
        assert guidance in tool.description


def test_chunk_context_publishes_governed_contract(settings_factory):
    tool = next(tool for tool in framework_tools.get_local_tools(settings_factory())
                if tool.name == "get_chunk_context")
    spec = tool.project_spec
    assert spec.args_model is GetChunkContextArgs
    assert spec.output_model is GetChunkContextOutput
    assert spec.version == tool.metadata["tool_version"] == "1.0.0"
    assert spec.risk_level is ToolRiskLevel.READ_ONLY
    assert spec.required_permissions == ("knowledge_base:read",)
    assert spec.timeout == 5.0 and spec.max_retries == 0
    assert spec.idempotent is True and spec.requires_approval is False
    assert spec.audit_output_mode is ToolAuditOutputMode.NONE
    schema = tool.tool_call_schema.model_json_schema()["properties"]
    assert set(schema) == {"chunk_id", "before", "after"}
    for field in ("before", "after"):
        assert schema[field]["minimum"] == 0 and schema[field]["maximum"] == 3
        assert schema[field]["default"] == 1
    for guidance in ("Document.metadata.chunk_id", "不能使用 Document.id", "不会跨文档", "不要机械", "evidence_ids"):
        assert guidance in spec.description


def test_catalog_caches_configuration_and_preserves_order(settings_factory):
    settings = settings_factory()
    changed = settings_factory(agent_max_total_cost_ms=60_000)
    specs = catalog.get_local_tool_specs(settings)
    tools = framework_tools.get_local_tools(settings)
    assert isinstance(specs, tuple) and isinstance(tools, tuple)
    assert specs is catalog.get_local_tool_specs(settings)
    assert tools is framework_tools.get_local_tools(settings)
    expected = (
        "search_knowledge_base", "find_documents", "search_document",
        "web_search", "web_fetch", "web_pdf_fetch",
        "compare_documents", "list_documents", "get_chunk_text", "get_chunk_context",
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
    assert names[2] == ["search_knowledge_base", "find_documents", "search_document", "compare_documents", "list_documents", "get_chunk_text", "get_chunk_context"]


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
