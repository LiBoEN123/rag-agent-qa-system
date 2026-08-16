"""请求级工具可见性与 Runtime 最终授权边界测试。"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
import app.tools.runtime as tool_runtime
from app.tools.context import ToolInvocationContext, has_permission
from app.tools.contracts import ToolRiskLevel, ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.providers.base import is_tool_visible
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success
from app.tools.runtime import execute_registered_tool
from tests.support.agent_runtime import build_agent_harness


class TextArgs(BaseModel):
    text: str


class VisibilityModel:
    """按序返回消息，并记录框架真实绑定的工具。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.received_messages.append(list(messages))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def isolate_tool_runtime():
    tool_runtime.reset_tool_runtime()
    yield
    tool_runtime.reset_tool_runtime()


def _spec(
    name: str,
    calls: list,
    *,
    required_permissions: tuple[str, ...] = (),
    description: str = "受控可见性测试工具",
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY,
    requires_approval: bool = False,
) -> ToolSpec:
    def handler(args, context):
        calls.append((name, dict(args), context))
        return tool_success(name, args, {"text": args["text"]})

    return ToolSpec(
        name=name,
        version="1.0.0",
        description=description,
        args_schema={"text": "string"},
        output_schema={"text": "string"},
        handler=handler,
        args_model=TextArgs,
        risk_level=risk_level,
        required_permissions=required_permissions,
        requires_approval=requires_approval,
    )


def _call(name: str, text: str, call_id: str) -> dict:
    return {
        "name": name,
        "args": {"text": text},
        "id": call_id,
        "type": "tool_call",
    }


def _invoke_graph(monkeypatch, registry, model, context):
    monkeypatch.setattr(framework_agent, "get_agent_chat_model", lambda: model)
    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    return build_agent_harness().invoke(
        {
            "original_question": "验证权限感知工具可见性",
            "effective_question": "验证权限感知工具可见性",
            "top_k": 5,
            "decompose_top_k": 10,
            "decision": {"action": "agent", "router_llm_used": False},
            "trace_id": context.trace_id,
            "session_id": context.session_id,
        },
        context=context,
    )


def test_public_and_single_permission_visibility_is_exact():
    public = _spec("public_tool", [])
    private = _spec(
        "private_tool", [], required_permissions=("documents:read",)
    )
    assert is_tool_visible(public, context=ToolInvocationContext()) is True
    assert is_tool_visible(private, context=ToolInvocationContext()) is False
    assert is_tool_visible(
        private,
        context=ToolInvocationContext(permissions=("documents:read",)),
    ) is True
    assert has_permission(("Documents:read",), "documents:read") is False
    with pytest.raises(ValueError, match="resource:action"):
        ToolInvocationContext(permissions=("Documents:read",))


def test_multiple_permissions_use_and_semantics():
    spec = _spec(
        "export_tool",
        [],
        required_permissions=("documents:read", "documents:export"),
    )
    assert is_tool_visible(
        spec,
        context=ToolInvocationContext(permissions=("documents:read",)),
    ) is False
    assert is_tool_visible(
        spec,
        context=ToolInvocationContext(
            permissions=("documents:read", "documents:export")
        ),
    ) is True


def test_provider_filters_in_order_without_mutation_or_execution():
    registry = ToolRegistry()
    calls: list = []
    public = _spec("public_tool", calls)
    hidden = _spec("hidden_tool", calls, required_permissions=("hidden:read",))
    allowed = _spec(
        "allowed_tool", calls, required_permissions=("documents:read",)
    )
    for spec in (public, hidden, allowed):
        registry.register(spec)
    context = ToolInvocationContext(permissions=("documents:read",))
    before = (context, public.required_permissions, hidden.required_permissions)
    specs = LocalToolProvider(registry).list_specs(context=context)
    assert specs == [public, allowed]
    assert (context, public.required_permissions, hidden.required_permissions) == before
    assert calls == []


def test_execution_policy_fields_do_not_hide_authorized_tool():
    spec = _spec(
        "approval_tool",
        [],
        required_permissions=("documents:write",),
        risk_level=ToolRiskLevel.INTERNAL_WRITE,
        requires_approval=True,
    )
    context = ToolInvocationContext(permissions=("documents:write",))
    assert is_tool_visible(spec, context=context) is True


def test_same_provider_keeps_request_visibility_isolated():
    registry = ToolRegistry()
    registry.register(_spec("public_tool", []))
    registry.register(
        _spec("private_tool", [], required_permissions=("documents:read",))
    )
    provider = LocalToolProvider(registry)

    contexts = (
        ToolInvocationContext(),
        ToolInvocationContext(permissions=("documents:read",)),
    )
    names = [
        [spec.name for spec in provider.list_specs(context=context)]
        for context in contexts
    ]
    assert names == [["public_tool"], ["public_tool", "private_tool"]]
    assert not hasattr(provider, "context")


@pytest.mark.parametrize(
    ("permissions", "has_search", "has_fetch"),
    [
        (("knowledge_base:read",), False, False),
        (("knowledge_base:read", "web:search"), True, False),
        (("knowledge_base:read", "web:fetch"), False, True),
        (
            ("knowledge_base:read", "web:search", "web:fetch"),
            True,
            True,
        ),
    ],
)
def test_production_web_permissions_are_independent(
    permissions, has_search, has_fetch, tool_registry
):
    """同一生产 Provider 连续请求保持 Search 与 Fetch 权限隔离。"""
    names = {
        spec.name
        for spec in LocalToolProvider(tool_registry).list_specs(
            context=ToolInvocationContext(permissions=permissions)
        )
    }

    assert ("web_search" in names) is has_search
    assert ("web_fetch" in names) is has_fetch
    assert ("web_pdf_fetch" in names) is has_fetch


def test_agent_binds_visible_tools_and_executes_authorized_tool(monkeypatch):
    registry = ToolRegistry()
    calls: list = []
    registry.register(_spec("public_tool", calls))
    registry.register(
        _spec("private_tool", calls, required_permissions=("documents:read",))
    )
    model = VisibilityModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("private_tool", "allowed", "private-call")],
            ),
            AIMessage(content="authorized tool complete"),
        ]
    )
    context = ToolInvocationContext(
        trace_id="trace-visible",
        principal_id="reader",
        permissions=("documents:read",),
        source="visibility_test",
    )
    runtime_calls = []
    original_execute = tool_runtime.ToolRuntime.execute

    def recording_execute(self, name, args=None, *, context=None):
        runtime_calls.append((name, context))
        return original_execute(self, name, args, context=context)

    monkeypatch.setattr(tool_runtime.ToolRuntime, "execute", recording_execute)
    output = _invoke_graph(monkeypatch, registry, model, context)
    assert model.bound_tool_names == ["public_tool", "private_tool"]
    assert len(runtime_calls) == 1
    assert runtime_calls[0][0] == "private_tool"
    execution_context = runtime_calls[0][1]
    assert execution_context.trace_id == context.trace_id
    assert execution_context.permissions == context.permissions
    assert execution_context.effective_question
    assert calls == [("private_tool", {"text": "allowed"}, execution_context)]
    assert output["result"]["answer"] == "authorized tool complete"


def test_forged_hidden_tool_is_unknown_without_metadata_leak(monkeypatch):
    registry = ToolRegistry()
    calls: list = []
    secret_description = "SECRET CAPABILITY DESCRIPTION"
    registry.register(_spec("public_tool", calls))
    registry.register(
        _spec(
            "private_tool",
            calls,
            required_permissions=("documents:read",),
            description=secret_description,
        )
    )
    model = VisibilityModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("private_tool", "forged", "hidden-call")],
            ),
            AIMessage(content="hidden tool handled safely"),
        ]
    )
    context = ToolInvocationContext(trace_id="trace-hidden")

    def forbidden_execute(*args, **kwargs):
        raise AssertionError("hidden tool must not reach ToolRuntime")

    monkeypatch.setattr(tool_runtime.ToolRuntime, "execute", forbidden_execute)
    output = _invoke_graph(monkeypatch, registry, model, context)
    message = model.received_messages[1][-1]
    payload = json.loads(message.content)
    assert model.bound_tool_names == ["public_tool"]
    assert isinstance(message, ToolMessage)
    assert payload["error_code"] == "unknown_tool"
    assert "documents:read" not in message.content
    assert secret_description not in message.content
    assert calls == []
    assert output["result"]["answer"] == "hidden tool handled safely"


def test_direct_execution_entry_remains_runtime_authorized():
    registry = ToolRegistry()
    calls: list = []
    registry.register(
        _spec("private_tool", calls, required_permissions=("documents:read",))
    )

    result = execute_registered_tool(
        registry=registry,
        tool_name="private_tool",
        tool_args={"text": "blocked"},
        context=ToolInvocationContext(trace_id="trace-direct"),
    )
    assert result["error_code"] == "permission_denied"
    assert result["governance"]["missing_permissions"] == ["documents:read"]
    assert calls == []
