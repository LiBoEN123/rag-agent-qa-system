"""工具声明、调用上下文与输入输出模型的契约校验。"""

import json
from dataclasses import fields
from typing import Any

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError

from app.agent.framework_tools import local_tool_contract
from app.contracts.chat import ToolMode
from app.tools.context import PrincipalType, ToolInvocationContext
from app.tools.contracts import (
    ToolSpec,
)
from app.tools.schemas import (
    GetChunkContextArgs,
    SearchDocumentOutput,
    SearchKnowledgeBaseArgs,
    SearchKnowledgeBaseOutput,
)


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
    )

    assert context.trace_id == "trace-1"
    assert context.session_id == "session-1"
    assert context.principal_id == "service-1"
    assert context.source == "test"
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
        "effective_question",
        "tool_mode",
    }
    assert context.tool_mode is ToolMode.FULL
    assert ToolInvocationContext(tool_mode="rag_only").tool_mode is ToolMode.RAG_ONLY


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("permissions", "knowledge_base:read"),
        ("permissions", ("knowledge_base:*",)),
        ("tool_mode", "invalid"),
        ("tool_mode", None),
    ],
)
def test_context_rejects_invalid_permissions(field_name, value):
    with pytest.raises(ValueError):
        ToolInvocationContext(**{field_name: value})


def test_search_knowledge_base_args_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=0)

    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query="测试问题", top_k=11)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_knowledge_base_args_rejects_blank_query(query):
    """空字符串和纯空白查询都在参数模型边界拒绝。"""
    with pytest.raises(ValidationError):
        SearchKnowledgeBaseArgs(query=query)


def test_search_knowledge_base_args_normalizes_query_and_top_k():
    """查询 trim 与数字字符串转换只由参数模型负责。"""
    args = SearchKnowledgeBaseArgs(query="  测试问题  ", top_k="5")

    assert args.model_dump(mode="json") == {"query": "测试问题", "top_k": 5}


def test_get_chunk_context_args_parses_positive_integer_locator():
    assert GetChunkContextArgs(chunk_id=1).chunk_id == 1
    assert GetChunkContextArgs(chunk_id=382).chunk_id == 382
    assert GetChunkContextArgs(chunk_id="382").chunk_id == 382


@pytest.mark.parametrize("chunk_id", [0, -1, "", "   ", "abc", "knowledge_base:ef463ed6"])
def test_get_chunk_context_args_rejects_invalid_locator(chunk_id):
    """拒绝非正整数与 evidence identity。"""
    with pytest.raises(ValidationError):
        GetChunkContextArgs(chunk_id=chunk_id)


def test_knowledge_output_contains_only_retrieval_evidence():
    schema = SearchKnowledgeBaseOutput.model_json_schema()
    assert set(schema["properties"]) == {"query", "count", "evidence"}
    assert "RAGResult" not in json.dumps(schema)


@pytest.mark.parametrize("field", ["rag_result", "answer", "references", "chunks", "top_chunk_ids", "top_scores"])
def test_knowledge_output_rejects_answer_contract(field):
    with pytest.raises(ValueError):
        SearchKnowledgeBaseOutput.model_validate({
            "query": "test", "count": 0, "evidence": [], field: {},
        })


def test_knowledge_output_keeps_count_consistency():
    with pytest.raises(ValueError):
        SearchKnowledgeBaseOutput.model_validate({
            "query": "test", "count": 1, "evidence": [],
        })


@pytest.mark.parametrize("field,value", [
    ("id", None), ("id", "  "), ("page_content", ""), ("page_content", " \n"),
    ("source_type", None), ("source_type", " "), ("source_uri", None),
    ("source_uri", " "), ("content_is_untrusted", False),
    ("content_is_untrusted", 1), ("content_is_untrusted", "true"),
])
@pytest.mark.parametrize("output_model", [SearchKnowledgeBaseOutput, SearchDocumentOutput])
def test_evidence_rejects_missing_identity_body_or_trust_metadata(field, value, output_model):
    document = Document(id="test", page_content="正文", metadata={
        "source_type": "knowledge_base", "source_uri": "test://1",
        "content_is_untrusted": True,
    }).model_dump(mode="json")
    target = document if field in {"id", "page_content"} else document["metadata"]
    target[field] = value
    with pytest.raises(ValueError, match=field):
        output_model.model_validate({
            "query": "test", "count": 1, "evidence": [document],
            **({"document": "source-a", "source_id": "source-a"}
               if output_model is SearchDocumentOutput else {}),
        })


def test_document_uses_native_pydantic_json_transport():
    document = Document(id="test", page_content="原始正文", metadata={
        "source_type": "knowledge_base", "source_uri": "test://1",
        "content_is_untrusted": True,
    })
    output = SearchKnowledgeBaseOutput.model_validate({
        "query": "test", "count": 1, "evidence": [document],
    })
    assert isinstance(output.evidence[0], Document)
    dumped = output.model_dump(mode="json")
    assert json.loads(json.dumps(dumped))["evidence"] == [document.model_dump(mode="json")]


class EchoArgs(BaseModel):
    """回显工具参数。"""

    value: str = Field(..., description="需要回显的文本")
    repeat: int = Field(default=1, ge=1, description="回显次数")


class EchoOutput(BaseModel):
    """回显工具输出。"""

    value: str


def echo_handler(
    args: dict,
    context: ToolInvocationContext,
) -> dict:
    """返回输入值，供契约测试使用。"""
    return {
        "status": "success",
        "ok": True,
        "tool_name": "echo_tool",
        "tool_input": args,
        "output": {"value": args["value"]},
        "error": "",
    }


def _make_spec(**overrides: Any) -> ToolSpec:
    """构造可按字段覆盖的合法测试契约。"""
    values = {
        "name": "echo_tool",
        "version": "1.0.0",
        "description": "回显输入",
        "output_model": EchoOutput,
        "handler": echo_handler,
        "args_model": EchoArgs,
    }
    values.update(overrides)
    return ToolSpec(**values)


def test_tool_spec_normalizes_and_serializes_contract():
    """合法契约会被规范化，并可安全序列化给治理与检查端。"""
    spec = _make_spec(
        name="  echo_tool  ",
        version=" 1.0.0-beta.1 ",
        description="  回显输入  ",
        output_model=EchoOutput,
        required_permissions=(
            " knowledge_base:read ",
            "ticket:create",
            "knowledge_base:read",
        ),
        timeout=5,
    )

    assert spec.name == "echo_tool"
    assert spec.version == "1.0.0-beta.1"
    assert spec.description == "回显输入"
    assert spec.timeout == 5.0
    assert isinstance(spec.timeout, float)
    assert spec.required_permissions == (
        "knowledge_base:read",
        "ticket:create",
    )

    contract_spec = local_tool_contract(spec)
    json.dumps(contract_spec)

    assert "handler" not in contract_spec
    assert contract_spec["args_model"] == "EchoArgs"
    assert contract_spec["output_model"] == "EchoOutput"
    args_schema = contract_spec["args_schema"]
    assert set(args_schema["properties"]) == {"value", "repeat"}
    assert args_schema["properties"]["value"] == {
        "description": "需要回显的文本",
        "title": "Value",
        "type": "string",
    }
    assert args_schema["properties"]["repeat"]["default"] == 1
    assert args_schema["properties"]["repeat"]["description"] == "回显次数"
    assert args_schema["properties"]["repeat"]["type"] == "integer"
    assert args_schema["required"] == ["value"]
    assert contract_spec["output_schema"] == EchoOutput.model_json_schema()
    assert contract_spec["required_permissions"] == [
        "knowledge_base:read",
        "ticket:create",
    ]


def test_tool_spec_derives_fresh_deterministic_schemas():
    """重复导出模型派生的稳定副本，外部修改不影响工具契约。"""
    tool = _make_spec()

    first = local_tool_contract(tool)
    first["args_schema"]["properties"].clear()
    first["output_schema"]["properties"].clear()
    second = local_tool_contract(tool)

    assert set(second["args_schema"]["properties"]) == {"value", "repeat"}
    assert set(second["output_schema"]["properties"]) == {"value"}
    assert second == local_tool_contract(tool)
    json.dumps(second)


def test_tool_spec_rejects_manual_args_schema():
    """参数 Schema 只能从 args_model 派生，构造器不再接收手写副本。"""
    with pytest.raises(ValueError, match="args_schema"):
        _make_spec(args_schema={"value": "string"})


def test_tool_spec_requires_output_model_and_rejects_manual_schema():
    """所有工具使用输出模型，不能通过手写 schema 绕过输出校验。"""
    with pytest.raises(ValueError, match="output_model"):
        _make_spec(output_model=None)
    with pytest.raises(ValueError, match="output_model"):
        ToolSpec(**_make_spec().model_dump(exclude={"output_model"}))
    with pytest.raises(ValueError, match="output_schema"):
        _make_spec(output_schema={"value": "string"})


@pytest.mark.parametrize(
    "name",
    ["", "   ", "Echo Tool", "echo-tool", "2echo", "搜索工具"],
)
def test_tool_spec_rejects_invalid_name(name):
    """工具名必须使用受限的小写 snake_case 格式。"""
    with pytest.raises(ValueError, match="name"):
        _make_spec(name=name)


@pytest.mark.parametrize("description", ["", "   "])
def test_tool_spec_rejects_empty_description(description):
    """工具描述去除首尾空格后不能为空。"""
    with pytest.raises(ValueError, match="description"):
        _make_spec(description=description)


@pytest.mark.parametrize("version", ["", "v1", "latest", "version-2"])
def test_tool_spec_rejects_invalid_version(version):
    """工具版本必须遵循基础 SemVer。"""
    with pytest.raises(ValueError, match="version"):
        _make_spec(version=version)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("inf"), float("nan")],
)
def test_tool_spec_rejects_invalid_timeout(timeout):
    """超时声明必须是有限正数。"""
    with pytest.raises(ValueError, match="timeout"):
        _make_spec(timeout=timeout)


@pytest.mark.parametrize("max_retries", [-1, True, False])
def test_tool_spec_rejects_invalid_max_retries(max_retries):
    """重试次数必须是非负整数，并显式拒绝 bool。"""
    with pytest.raises(ValueError, match="max_retries"):
        _make_spec(max_retries=max_retries)


@pytest.mark.parametrize(
    "required_permissions",
    [
        "knowledge_base:read",
        ("",),
        ("   ",),
        ("Knowledge Base:Read",),
        ("read",),
    ],
)
def test_tool_spec_rejects_invalid_permissions(required_permissions):
    """权限必须以规范的字符串序列声明。"""
    with pytest.raises(
        ValueError,
        match="required_permissions",
    ):
        _make_spec(required_permissions=required_permissions)


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("output_schema", []),
        ("handler", None),
        ("args_model", object),
        ("output_model", object),
        ("idempotent", 1),
        ("requires_approval", 0),
        ("risk_level", "unknown"),
    ],
)
def test_tool_spec_rejects_invalid_base_field_types(
    field_name,
    invalid_value,
):
    """基础契约字段在创建阶段执行明确的类型校验。"""
    with pytest.raises(ValueError) as exc_info:
        _make_spec(**{field_name: invalid_value})

    error = str(exc_info.value)
    assert field_name in error

