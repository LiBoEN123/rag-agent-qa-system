"""Registry-driven Agent 的动态工具与请求隔离回归测试。"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from app.agent import framework_agent, framework_tools
from app.agent.framework_tools import build_framework_tools
from app.graph.builder import build_agent_graph
from app.tools.context import ToolInvocationContext
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.result import tool_success


class EchoTextArgs(BaseModel):
    """动态回显工具参数。"""

    text: str


class JoinTextsArgs(BaseModel):
    """动态拼接工具参数。"""

    parts: list[str]
    separator: str = Field(default=" ")


class FakeAgentModel:
    """按序返回消息并记录每次绑定到模型的动态工具。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []
        self.bound_tools = []
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        return self.responses.pop(0)


def _register_dynamic_tools(
    registry: ToolRegistry,
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

    registry.register(
        ToolSpec(
            name="echo_text",
            version="1.0.0",
            description="回显传入的 text，用于需要原样返回测试文本的场景",
            args_schema={"text": "string"},
            output_schema={"text": "string"},
            handler=echo_handler,
            args_model=EchoTextArgs,
        )
    )
    registry.register(
        ToolSpec(
            name="join_texts",
            version="1.0.0",
            description="按 separator 拼接 parts 列表并返回拼接文本",
            args_schema={"parts": "list[string]", "separator": "string"},
            output_schema={"text": "string"},
            handler=join_handler,
            args_model=JoinTextsArgs,
        )
    )


def _invoke_dynamic_graph(
    monkeypatch,
    *,
    registry: ToolRegistry,
    model: FakeAgentModel,
    context: ToolInvocationContext,
    question: str = "调用动态测试工具",
):
    """把独立 Provider 注入现有 Graph，不修改任何编排分支。"""
    provider = LocalToolProvider(registry)
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: provider,
    )
    graph_input = {
        "original_question": question,
        "effective_question": question,
        "top_k": 5,
        "decompose_top_k": 10,
        "trace_id": context.trace_id,
        "session_id": context.session_id,
    }
    return build_agent_graph().compile().invoke(
        graph_input,
        context=context,
    )


def test_tool_spec_builds_framework_structured_tool_from_provider():
    """唯一框架适配完整复用 ToolSpec 定义、Schema 与 Registry。"""
    registry = ToolRegistry()
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    context = ToolInvocationContext(
        trace_id="trace-adapter",
        session_id="session-adapter",
        principal_id="adapter-test",
        permissions=("knowledge_base:read",),
        source="registry_adapter_test",
    )
    provider = LocalToolProvider(registry)

    tool = next(
        item
        for item in build_framework_tools(provider, context=context)
        if item.name == "join_texts"
    )

    assert tool.name == "join_texts"
    assert tool.description == registry.get("join_texts").description
    assert issubclass(tool.args_schema, JoinTextsArgs)
    assert "runtime" not in tool.tool_call_schema.model_fields
    assert contexts == []


def test_agent_discovers_and_calls_echo_without_core_tool_mapping(monkeypatch):
    """注册 echo_text 后，现有 Agent Graph 可直接发现、调用并继续回答。"""
    registry = ToolRegistry()
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeAgentModel(
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
            AIMessage(content="echo complete"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-echo",
        session_id="session-echo",
        principal_id="echo-user",
        permissions=("knowledge_base:read",),
        source="registry_driven_test",
    )

    output = _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )

    assert model.bound_tool_names == ["echo_text", "join_texts"]
    message = model.received_messages[1][-1]
    assert isinstance(message, ToolMessage)
    assert json.loads(message.content)["output"] == {"text": "hello"}
    assert output["result"]["answer"] == "echo complete"
    assert contexts == [context]
    assert contexts[0].trace_id == "trace-echo"
    assert contexts[0].session_id == "session-echo"
    assert contexts[0].permissions == ("knowledge_base:read",)
    assert contexts[0].source == "registry_driven_test"
    assert tool_registry.get("echo_text") is None


def test_new_tool_request_reaches_agent_without_router_changes(monkeypatch):
    """新工具和新口吻经固定 route 入口后仍由 Agent 动态发现。"""
    registry = ToolRegistry()
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeAgentModel(
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
            AIMessage(content="任意新口吻"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-dynamic-route",
        source="registry_driven_route_test",
    )

    output = _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
        question="请用刚增加的处理能力原样接住这句话",
    )

    assert output["decision"]["action"] == "agent"
    assert "tool_name" not in output["decision"]
    assert model.bound_tool_names == ["echo_text", "join_texts"]
    assert output["result"]["answer"] == "任意新口吻"
    assert contexts == [context]


def test_agent_binds_different_dynamic_args_schema(monkeypatch):
    """动态发现不是单字符串特判，列表和默认参数 Schema 同样可执行。"""
    registry = ToolRegistry()
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeAgentModel(
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
            AIMessage(content="registry driven"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-join",
        session_id="session-join",
        source="registry_driven_test",
    )

    _invoke_dynamic_graph(
        monkeypatch,
        registry=registry,
        model=model,
        context=context,
    )

    tool = next(item for item in model.bound_tools if item.name == "join_texts")
    assert issubclass(tool.args_schema, JoinTextsArgs)
    assert "runtime" not in tool.tool_call_schema.model_fields
    payload = json.loads(model.received_messages[1][-1].content)
    assert payload["output"] == {"text": "registry driven"}


def test_unknown_model_tool_call_is_structured_and_recoverable(monkeypatch):
    """模型伪造未注册名称时不执行 callable，并可读取错误后继续决策。"""
    registry = ToolRegistry()
    contexts: list[ToolInvocationContext] = []
    _register_dynamic_tools(registry, contexts)
    model = FakeAgentModel(
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
            AIMessage(content="unknown tool handled"),
        ]
    )

    output = _invoke_dynamic_graph(
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


def test_provider_request_isolation(monkeypatch):
    """两个请求的独立 Registry、工具列表和 Context 不会相互泄漏。"""
    seen: list[tuple[str, str]] = []

    class RequestArgs(BaseModel):
        value: str

    def run_request(tool_name: str, trace_id: str):
        registry = ToolRegistry()

        def handler(args, context):
            seen.append((tool_name, context.trace_id))
            return tool_success(tool_name, args, {"value": args["value"]})

        registry.register(
            ToolSpec(
                name=tool_name,
                version="1.0.0",
                description=f"仅供 {trace_id} 使用的请求隔离工具",
                args_schema={"value": "string"},
                output_schema={"value": "string"},
                handler=handler,
                args_model=RequestArgs,
            )
        )
        model = FakeAgentModel(
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
                AIMessage(content=trace_id),
            ]
        )
        _invoke_dynamic_graph(
            monkeypatch,
            registry=registry,
            model=model,
            context=ToolInvocationContext(
                trace_id=trace_id,
                session_id=f"session-{trace_id}",
                source="isolation_test",
            ),
        )
        return model.bound_tool_names

    assert run_request("request_a_tool", "trace-a") == ["request_a_tool"]
    assert run_request("request_b_tool", "trace-b") == ["request_b_tool"]
    assert seen == [
        ("request_a_tool", "trace-a"),
        ("request_b_tool", "trace-b"),
    ]


def test_direct_dispatch_still_enforces_runtime_permissions():
    """绕过模型可见性直接分发时，最终权限仍由 ToolRuntime 拒绝。"""
    registry = ToolRegistry()
    called: list[str] = []

    def handler(args, context):
        called.append(args["text"])
        return tool_success("protected_echo", args, {"text": args["text"]})

    registry.register(
        ToolSpec(
            name="protected_echo",
            version="1.0.0",
            description="需要 dynamic:use 权限的测试回显工具",
            args_schema={"text": "string"},
            output_schema={"text": "string"},
            handler=handler,
            args_model=EchoTextArgs,
            required_permissions=("dynamic:use",),
        )
    )
    context = ToolInvocationContext(
        trace_id="trace-denied",
        source="permission_test",
    )
    result = registry.dispatch(
        "protected_echo",
        {"text": "blocked"},
        context=context,
    )

    assert result["ok"] is False
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["missing_permissions"] == ["dynamic:use"]
    assert called == []
