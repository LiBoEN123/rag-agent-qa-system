import pytest

from app import agent


def test_validate_llm_rag_route_normalizes_safe_args():
    """验证知识库路由会忽略不安全的模型参数并使用原问题。"""
    decision = agent.validate_route_decision(
        {
            "decision": "use_knowledge_base",
            "action": "search_knowledge_base",
            "tool_name": "some_hallucinated_tool",
            "tool_args": {"query": "bad rewrite", "top_k": 999},
            "reason": "需要查知识库",
            "confidence": 0.9,
        },
        "Vue 应用如何创建？",
        3,
    )

    assert decision["decision"] == "use_knowledge_base"
    assert decision["action"] == "search_knowledge_base"
    assert decision["tool_name"] == ""
    assert decision["tool_args"] == {"query": "Vue 应用如何创建?", "top_k": 3}
    assert decision["router_source"] == "llm"
    assert decision["router_confidence"] == 0.9


def test_validate_llm_tool_route_checks_calculator_expression():
    """验证计算器路由会规范化并校验表达式参数。"""
    decision = agent.validate_route_decision(
        {
            "decision": "use_tool",
            "action": "simple_calculator",
            "tool_name": "simple_calculator",
            "tool_args": {"expression": "１＋２*3"},
            "reason": "需要计算",
        },
        "帮我算 1 + 2 * 3",
        5,
    )

    assert decision["decision"] == "use_tool"
    assert decision["tool_name"] == "simple_calculator"
    assert decision["tool_args"] == {"expression": "1+2*3"}


def test_validate_llm_route_rejects_unknown_tool():
    """验证未知工具路由会被拒绝。"""
    with pytest.raises(ValueError):
        agent.validate_route_decision(
            {
                "decision": "use_tool",
                "action": "delete_documents",
                "tool_name": "delete_documents",
                "tool_args": {},
            },
            "删除所有文档",
            5,
        )


def test_route_uses_llm_when_enabled(monkeypatch):
    """验证启用 LLM 路由时会采用模型决策并清洗参数。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    monkeypatch.setattr(
        agent,
        "route_agent_action_by_llm",
        lambda question, top_k, trace_id="": {
            "decision": "use_tool",
            "action": "list_documents",
            "tool_name": "list_documents",
            "tool_args": {"ignored": True},
            "reason": "用户询问文档列表",
            "confidence": 0.8,
        },
    )

    decision = agent.route_agent_action("知识库有哪些文档？", 5)

    assert decision["action"] == "list_documents"
    assert decision["tool_args"] == {}
    assert decision["router_source"] == "llm"


def test_route_falls_back_to_rules_when_llm_is_invalid(monkeypatch):
    """验证 LLM 路由非法时会回退到规则路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", True, raising=False)
    monkeypatch.setattr(
        agent,
        "route_agent_action_by_llm",
        lambda question, top_k, trace_id="": {"action": "unknown_tool"},
    )

    decision = agent.route_agent_action("1 + 2 * 3", 5)

    assert decision["action"] == "simple_calculator"
    assert decision["router_source"] == "rule_based_fallback"
    assert "未知路由 action" in decision["router_fallback_reason"]


def test_route_uses_rules_when_router_disabled(monkeypatch):
    """验证关闭 LLM 路由时直接使用规则路由。"""
    monkeypatch.setattr(agent.Settings, "ENABLE_LLM_ROUTER", False, raising=False)

    decision = agent.route_agent_action("知识库有多少文档？", 5)

    assert decision["action"] == "list_documents"
    assert decision["router_source"] == "rule_based"
