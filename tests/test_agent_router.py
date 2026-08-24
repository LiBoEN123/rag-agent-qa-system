import inspect

import pytest
from pydantic import BaseModel

from app.agent.router import route_agent_action
from app.llm import factory as model_factory
from app.llm.gateway import ModelGateway
from app.tools.contracts import ToolSpec


def test_empty_question_returns_clarify_without_router_model(monkeypatch):
    """空问题由确定性入口要求澄清，不产生 Router 模型调用。"""

    def fail_model_call(*args, **kwargs):
        raise AssertionError("确定性 Router 不应调用模型")

    monkeypatch.setattr(ModelGateway, "invoke_text", fail_model_call)
    monkeypatch.setattr(model_factory, "_create_model", fail_model_call)

    decision = route_agent_action(" \n ")

    assert decision["decision"] == "clarify"
    assert decision["action"] == "clarify"
    assert decision["clarification_question"]
    assert decision["router_source"] == "rule_based_guardrail"
    assert decision["router_confidence"] == 1.0


@pytest.mark.parametrize(
    "question",
    ["你是谁？", "你能做什么？", "你可以做什么？", "你有什么功能？"],
)
def test_capability_questions_return_direct_answer(question):
    """少量稳定的系统能力问题继续走 direct_answer。"""
    decision = route_agent_action(question)

    assert decision["decision"] == "direct_answer"
    assert decision["action"] == "direct_answer"
    assert decision["router_source"] == "rule_based"


@pytest.mark.parametrize(
    "question",
    ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"],
)
def test_ambiguous_references_return_clarify(question):
    """少量明确指代不明的问题继续走 clarify。"""
    decision = route_agent_action(question)

    assert decision["decision"] == "clarify"
    assert decision["action"] == "clarify"
    assert "具体指" in decision["clarification_question"]
    assert decision["router_source"] == "rule_based"


@pytest.mark.parametrize(
    "question",
    [
        "知识库有哪些文档？",
        "请对比《学生手册》和《调整通知》的差异。",
        "12 * 8",
        "查看切块 5 的内容",
        "搜索一下最新公开资料",
        "读取 https://example.com/page",
        "调用刚注册的新能力处理这段输入",
    ],
)
def test_normal_questions_enter_agent_without_concrete_tool_choice(question):
    """普通问题统一进入 Agent，Router 不选择工具或构造参数。"""
    decision = route_agent_action(question)

    assert decision["decision"] == "use_agent"
    assert decision["action"] == "agent"
    assert decision["router_source"] == "rule_based"
    assert decision["router_confidence"] == 1.0
    assert set(decision) == {
        "decision",
        "action",
        "reason",
        "router_source",
        "router_confidence",
    }


def test_registered_tool_does_not_require_router_change(monkeypatch, tool_registry):
    """新增 Registry Tool 后，确定性 Router 仍返回同一通用 Agent 入口。"""

    class NewToolArgs(BaseModel):
        value: str = ""

    monkeypatch.setattr(tool_registry, "_tools", dict(tool_registry._tools))
    before = route_agent_action("调用刚注册的新能力处理这段输入")
    tool_registry.register(
        ToolSpec(
            name="pr9_test_tool",
            version="1.0.0",
            description="仅用于验证新增工具不要求修改 Router。",
            output_schema={"value": "string"},
            args_model=NewToolArgs,
            handler=lambda args, context: {"value": args.get("value", "")},
        )
    )
    after = route_agent_action("调用刚注册的新能力处理这段输入")

    assert "pr9_test_tool" in tool_registry.names()
    assert after == before
    assert after["action"] == "agent"


def test_router_signature_contains_only_question():
    """Router 不接受工具参数、trace 元数据或外部 decision。"""
    signature = inspect.signature(route_agent_action)

    assert list(signature.parameters) == ["question"]
