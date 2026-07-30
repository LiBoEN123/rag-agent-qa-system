"""Agent 的确定性粗粒度路由决策。"""

from typing import Literal, NotRequired, TypedDict

from app.agent.rules import normalize_question


RouteAction = Literal["agent", "direct_answer", "clarify"]
RouteDecisionType = Literal["use_agent", "direct_answer", "clarify"]


class RouteDecision(TypedDict):
    """父图 Router 写入 Graph State 的最小内部协议。"""

    decision: RouteDecisionType
    action: RouteAction
    reason: str
    clarification_question: NotRequired[str]
    router_source: Literal["rule_based", "rule_based_guardrail"]
    router_confidence: float


def route_agent_action(question: str) -> RouteDecision:
    """执行不依赖工具、模型或运行时上下文的确定性入口分流。"""
    normalized_question = normalize_question(question)

    if not normalized_question:
        return {
            "decision": "clarify",
            "action": "clarify",
            "reason": "用户未输入问题",
            "clarification_question": (
                "请直接输入你想查询的具体问题，例如："
                "每个 Vue 应用是通过什么函数创建实例的？"
            ),
            "router_source": "rule_based_guardrail",
            "router_confidence": 1.0,
        }

    if normalized_question in [
        "你是谁?",
        "你能做什么?",
        "你可以做什么?",
        "你有什么功能?",
    ]:
        return {
            "decision": "direct_answer",
            "action": "direct_answer",
            "reason": "用户询问系统能力，可以直接回答",
            "router_source": "rule_based",
            "router_confidence": 1.0,
        }

    if normalized_question in [
        "这个怎么弄",
        "这个怎么配置",
        "这个是什么意思",
        "这个怎么解决",
    ]:
        return {
            "decision": "clarify",
            "action": "clarify",
            "reason": "问题存在明显指代不明",
            "clarification_question": (
                "你说的“这个”具体指哪个功能、哪份文档或哪段报错？"
                "请补充一下对象。"
            ),
            "router_source": "rule_based",
            "router_confidence": 1.0,
        }

    return {
        "decision": "use_agent",
        "action": "agent",
        "reason": "交给 Agent 根据当前请求可见的动态工具定义决策",
        "router_source": "rule_based",
        "router_confidence": 1.0,
    }


__all__ = ["RouteDecision", "route_agent_action"]
