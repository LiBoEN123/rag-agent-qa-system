"""Agent 路由决策模块。

正常请求只做安全的粗粒度分流；具体工具意图统一交给动态 Agent。
"""

from app.agent.rules import normalize_question


# =========================
# Agent 路由决策
# =========================

def add_router_metadata(
        decision: dict,
        source: str,
        confidence: float | None = None,
        fallback_reason: str = "",
        llm_used: bool | None = None
) -> dict:
    """给路由决策补充来源、置信度和回退原因等元数据。"""
    decision = dict(decision or {})
    decision["router_source"] = source

    if confidence is not None:
        decision["router_confidence"] = confidence

    if llm_used is None:
        llm_used = source in {"llm", "rule_based_fallback"}
    decision["router_llm_used"] = bool(llm_used)

    if fallback_reason:
        decision["router_fallback_reason"] = fallback_reason[:300]

    decision.setdefault("router_fallback_reason", "")

    return decision


def route_agent_action_rule_based(question: str, top_k: int) -> dict:
    """只处理安全入口；其余表达统一交给动态 Agent。"""
    q = normalize_question(question)

    # 1. 空问题：需要澄清
    if not q:
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "用户未输入问题",
            "clarification_question": "请直接输入你想查询的具体问题，例如：每个 Vue 应用是通过什么函数创建实例的？"
        }

    # 2. 用户问系统能力：可以直接回答，不需要查知识库
    if q in ["你是谁?", "你能做什么?", "你可以做什么?", "你有什么功能?"]:
        return {
            "decision": "direct_answer",
            "action": "direct_answer",
            "tool_name": "",
            "tool_args": {},
            "reason": "用户询问系统能力，可以直接回答"
        }

    # 3. 指代太模糊：澄清
    if q in ["这个怎么弄", "这个怎么配置", "这个是什么意思", "这个怎么解决"]:
        return {
            "decision": "clarify",
            "action": "clarify",
            "tool_name": "",
            "tool_args": {},
            "reason": "问题存在明显指代不明",
            "clarification_question": "你说的“这个”具体指哪个功能、哪份文档或哪段报错？请补充一下对象。"
        }

    # 4. 默认进入 Agent。静态 Router 不猜测动态 Provider 中是否存在适用工具。
    return {
        "decision": "use_agent",
        "action": "agent",
        "tool_name": "",
        "tool_args": {},
        "reason": "交给 Agent 根据当前请求可见的动态工具定义决策",
    }


def route_agent_action(question: str, top_k: int, trace_id: str = "") -> dict:
    """执行不依赖具体工具名称或用户口吻的安全入口分流。"""
    q = normalize_question(question)
    rule_decision = route_agent_action_rule_based(question, top_k)

    return add_router_metadata(
        rule_decision,
        source="rule_based_guardrail" if not q else "rule_based",
        confidence=1.0,
        llm_used=False,
    )
