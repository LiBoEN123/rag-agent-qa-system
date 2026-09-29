"""Registry-driven Agent 的动态工具与请求隔离回归测试。"""

from pydantic import RootModel

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.agent import framework_agent, framework_tools
from tests.support.local_tools import invoke_local_tool
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from tests.support.local_tools import local_tools
from app.tools.catalog import get_local_tool_specs
from app.tools.result import tool_success
from app.agent.final_response import FinalAgentResponse
from tests.support.agent_runtime import FakeToolCallingModel, build_agent_harness
from app.agent.runtime import reset_agent_runtime


class EchoTextArgs(BaseModel):
    """动态回显工具参数。"""

    text: str


class JoinTextsArgs(BaseModel):
    """动态拼接工具参数。"""

    parts: list[str]
    separator: str = Field(default=" ")


def _invoke_local(
    registry: list[ToolSpec],
    tool_name: str,
    tool_args: dict,
    context: ToolInvocationContext,
):
    """通过 BaseTool.ainvoke 覆盖非 Agent 诊断调用。"""
    return invoke_local_tool(registry, tool_name, tool_args, context=context)


def _register_dynamic_tools(
    registry: list[ToolSpec],
    seen_contexts: list[ToolInvocationContext],
) -> None:
    """只向测试 Registry 注册两种不同参数结构的工具。"""

    def echo_handler(args, context):
        seen_contexts.append(context)
        return tool_success(
            "echo_text",
            args,
            {"text": args["text"]},
        )

    def join_handler(args, context):
        seen_contexts.append(context)
        return tool_success(
            "join_texts",
            args,
            {"text": args["separator"].join(args["parts"])},
        )

    registry.append(
        ToolSpec(
            name="echo_text",
            version="1.0.0",
            description="回显传入的 text，用于需要原样返回测试文本的场景",
            output_model=RootModel[dict],
            handler=echo_handler,
            args_model=EchoTextArgs,
        )
    )
    registry.append(
        ToolSpec(
            name="join_texts",
            version="1.0.0",
            description="按 separator 拼接 parts 列表并返回拼接文本",
            output_model=RootModel[dict],
            handler=join_handler,
            args_model=JoinTextsArgs,
        )
    )


async def _invoke_dynamic_graph(
    monkeypatch,
    *,
    registry: list[ToolSpec],
    model: FakeToolCallingModel,
    context: ToolInvocationContext,
    question: str = "调用动态测试工具",
):
    """把独立 Provider 注入现有 Graph，不修改任何编排分支。"""
    provider = local_tools(registry)
    reset_agent_runtime()
    monkeypatch.setattr(framework_agent, "get_answer_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_local_tools",
        lambda settings: provider,
    )
    graph_input = {
        "original_question": question,
        "effective_question": question,
        "trace_id": context.trace_id,
        "session_id": context.session_id,
    }
    return [state async for state in build_agent_harness().astream(
        graph_input,
        context=context,
    )][-1]


def test_tool_spec_builds_framework_structured_tool_from_provider():
    """自定义工具的模型同时驱动 Registry、StructuredTool 与 Runtime。"""
    registry = []
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    context = ToolInvocationContext(
        trace_id="trace-adapter",
        session_id="session-adapter",
        principal_id="adapter-test",
        permissions=("knowledge_base:read",),
        source="registry_adapter_test",
    )
    provider = local_tools(registry)

    tool = next(
        item
        for item in provider
        if item.name == "join_texts"
    )

    assert tool.name == "join_texts"
    assert tool.description == next((item for item in registry if item.name == "join_texts"), None).description
    assert tool.args_schema == JoinTextsArgs.model_json_schema()
    assert "runtime" not in tool.tool_call_schema["properties"]
    join_spec = next(item for item in registry if item.name == "join_texts")
    exported = framework_tools.local_tool_contract(join_spec)
    framework_schema = tool.tool_call_schema
    assert exported["args_schema"]["properties"] == framework_schema["properties"]
    assert exported["args_schema"]["required"] == framework_schema["required"]
    assert exported["args_schema"]["properties"]["separator"]["default"] == " "
    assert framework_tools.local_tool_contract(join_spec) == exported
    json.dumps(exported)

    invalid = invoke_local_tool(registry, "join_texts", {"parts": "不是列表"}, context=context)
    assert invalid["error_code"] == "invalid_arguments"
    assert contexts == []

    valid = invoke_local_tool(registry, "join_texts", {"parts": ["a", "b"]}, context=context)
    assert valid["output"] == {"text": "a b"}
    assert contexts == [context]


@pytest.mark.asyncio
async def test_agent_discovers_and_calls_echo_without_core_tool_mapping(monkeypatch):
    """注册 echo_text 后，现有 Agent Graph 可直接发现、调用并继续回答。"""
    registry = []
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_text",
                        "args": {"text": "hello"},
                        "id": "echo-call",
                        "type": "tool_call",
                    }
                ],
            ),
            FinalAgentResponse(outcome="answer", answer="echo complete"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-echo",
        session_id="session-echo",
        principal_id="echo-user",
        permissions=("knowledge_base:read",),
        source="registry_driven_test",
    )

    output = await _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )

    assert model.bound_tool_names == ["echo_text", "join_texts", model.structured_tool_name]
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage)
    assert (message.artifact or json.loads(message.content))["output"] == {"text": "hello"}
    assert output["result"]["answer"] == "echo complete"
    assert len(contexts) == 1
    assert contexts[0].trace_id == "trace-echo"
    assert contexts[0].session_id == "session-echo"
    assert contexts[0].permissions == ("knowledge_base:read",)
    assert contexts[0].source == "registry_driven_test"
    assert contexts[0].effective_question == "调用动态测试工具"
    assert next((item for item in get_local_tool_specs(get_settings()) if item.name == "echo_text"), None) is None


@pytest.mark.asyncio
async def test_new_tool_request_reaches_agent_without_router_changes(monkeypatch):
    """新工具和新口吻经固定 route 入口后仍由 Agent 动态发现。"""
    registry = []
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_text",
                        "args": {"text": "任意新口吻"},
                        "id": "dynamic-route-call",
                        "type": "tool_call",
                    }
                ],
            ),
            FinalAgentResponse(outcome="answer", answer="任意新口吻"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-dynamic-route",
        source="registry_driven_route_test",
    )

    output = await _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
        question="请用刚增加的处理能力原样接住这句话",
    )

    assert output["result"]["action"] == "agent"
    assert model.bound_tool_names == ["echo_text", "join_texts", model.structured_tool_name]
    assert output["result"]["answer"] == "任意新口吻"
    assert len(contexts) == 1
    assert contexts[0].trace_id == context.trace_id
    assert contexts[0].effective_question == "请用刚增加的处理能力原样接住这句话"


@pytest.mark.asyncio
async def test_agent_binds_different_dynamic_args_schema(monkeypatch):
    """动态发现不是单字符串特判，列表和默认参数 Schema 同样可执行。"""
    registry = []
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "join_texts",
                        "args": {"parts": ["registry", "driven"]},
                        "id": "join-call",
                        "type": "tool_call",
                    }
                ],
            ),
            FinalAgentResponse(outcome="answer", answer="registry driven"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-join",
        session_id="session-join",
        source="registry_driven_test",
    )

    await _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )

    tool = next(item for item in model.bound_tools if item.name == "join_texts")
    assert tool.args_schema == JoinTextsArgs.model_json_schema()
    assert "runtime" not in tool.tool_call_schema["properties"]
    payload = json.loads(model.received_messages[1][-1].content)
    assert payload["output"] == {"text": "registry driven"}


@pytest.mark.asyncio
async def test_unknown_model_tool_call_is_structured_and_recoverable(monkeypatch):
    """模型伪造未注册名称时不执行 callable，并可读取错误后继续决策。"""
    registry = []
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeToolCallingModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "not_registered",
                        "args": {"text": "unsafe"},
                        "id": "unknown-call",
                        "type": "tool_call",
                    }
                ],
            ),
            FinalAgentResponse(outcome="answer", answer="unknown tool handled"),
        ]
    )

    output = await _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=ToolInvocationContext(
            trace_id="trace-unknown",
            source="registry_driven_test",
        ),
    )

    payload = json.loads(model.received_messages[1][-1].content)
    assert payload["ok"] is False
    assert payload["error_code"] == "unknown_tool"
    assert contexts == []
    assert output["result"]["answer"] == "unknown tool handled"


@pytest.mark.asyncio
async def test_provider_request_isolation(monkeypatch):
    """两个请求的独立 Registry、工具列表和 Context 不会相互泄漏。"""
    seen: list[tuple[str, str]] = []

    class RequestArgs(BaseModel):
        value: str

    async def run_request(tool_name: str, trace_id: str):
        registry = []

        def handler(args, context):
            seen.append((tool_name, context.trace_id))
            return tool_success(tool_name, args, {"value": args["value"]})

        registry.append(
            ToolSpec(
                name=tool_name,
                version="1.0.0",
                description=f"仅供 {trace_id} 使用的请求隔离工具",
                output_model=RootModel[dict],
                handler=handler,
                args_model=RequestArgs,
            )
        )
        model = FakeToolCallingModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"value": trace_id},
                            "id": f"{trace_id}-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                FinalAgentResponse(outcome="answer", answer=trace_id),
            ]
        )
        await _invoke_dynamic_graph(
            monkeypatch,
            registry=registry,
            model=model,
            context=ToolInvocationContext(
                trace_id=trace_id,
                session_id=f"session-{trace_id}",
                source="isolation_test",
            ),
        )
        assert model.structured_tool_name in model.bound_tool_names
        return [name for name in model.bound_tool_names if name != model.structured_tool_name]

    assert await run_request("request_a_tool", "trace-a") == ["request_a_tool"]
    assert await run_request("request_b_tool", "trace-b") == ["request_b_tool"]
    assert seen == [
        ("request_a_tool", "trace-a"),
        ("request_b_tool", "trace-b"),
    ]


def test_direct_execution_entry_still_enforces_runtime_permissions():
    """绕过模型可见性直接执行时，最终权限仍由项目策略拒绝。"""
    registry = []
    called: list[str] = []

    def handler(args, context):
        called.append(args["text"])
        return tool_success("protected_echo", args, {"text": args["text"]})

    registry.append(
        ToolSpec(
            name="protected_echo",
            version="1.0.0",
            description="需要 dynamic:use 权限的测试回显工具",
            output_model=RootModel[dict],
            handler=handler,
            args_model=EchoTextArgs,
            required_permissions=("dynamic:use",),
        )
    )
    context = ToolInvocationContext(
        trace_id="trace-denied",
        source="permission_test",
    )
    result = _invoke_local(
        registry,
        "protected_echo",
        {"text": "blocked"},
        context,
    )

    assert result["ok"] is False
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["missing_permissions"] == ["dynamic:use"]
    assert called == []
