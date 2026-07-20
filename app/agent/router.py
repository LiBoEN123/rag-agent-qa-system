"""Agent 路由决策模块。

正常请求只做安全的粗粒度分流；具体工具意图统一交给动态 Agent。
"""

from app.agent.llm_router import (
    call_router_llm,
    parse_router_json_output,
)
from app.agent.rules import normalize_question
from app.tools.registry import tool_registry

# 仅供解析旧 Router 输出和兼容现有导入方；正常 Router Prompt 不读取具体工具。
ROUTER_ALLOWED_TOOLS = set(tool_registry.names())
ROUTER_TOOL_SPECS = tool_registry.router_specs()

ROUTER_ALLOWED_ACTIONS = {
    "agent",
    "rag_fast_path",
    "clarify",
    "direct_answer",
}

HIGH_CONFIDENCE_RULE_ACTIONS = {
    "agent",
    "direct_answer",
    "clarify",
}

ROUTER_ACTION_TO_DECISION = {
    "agent": "use_agent",
    "rag_fast_path": "use_knowledge_base",
    "clarify": "clarify",
    "direct_answer": "direct_answer",
}

DEFAULT_CLARIFICATION_QUESTION = "请补充你想查询的具体对象、文档、功能或报错信息。"


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


def route_agent_action_by_llm(question: str, top_k: int, trace_id: str = "") -> dict:
    """使用 LLM 路由并解析模型返回的决策。"""
    llm_output = call_router_llm(question, top_k, trace_id=trace_id)
    return parse_router_json_output(llm_output)


def normalize_router_confidence(value) -> float:
    """将路由置信度安全转换并限制在 0 到 1 之间。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, confidence))


def normalize_router_reason(raw_decision: dict) -> str:
    """提取并截断路由模型给出的决策理由。"""
    reason = str(raw_decision.get("reason") or "LLM 路由决策").strip()
    return reason[:200]


def normalize_chunk_id_value(chunk_id):
    """将工具参数里的 chunk_id 规整为整数或可查询字符串。"""
    if chunk_id is None:
        return None

    if isinstance(chunk_id, int):
        return chunk_id

    chunk_id_text = str(chunk_id).strip()
    if not chunk_id_text:
        return None

    return int(chunk_id_text) if chunk_id_text.isdigit() else chunk_id_text


def validate_route_decision(raw_decision: dict, question: str, top_k: int) -> dict:
    """校验 LLM 的粗粒度路由输出，不解析任何普通工具参数。"""
    if not isinstance(raw_decision, dict):
        raise ValueError("路由模型返回值不是 dict")

    q = normalize_question(question)
    if not q:
        return add_router_metadata(
            route_agent_action_rule_based(question, top_k),
            source="rule_based_guardrail"
        )

    raw_action = str(raw_decision.get("action") or "").strip()
    raw_decision_type = str(raw_decision.get("decision") or "").strip()

    if not raw_action:
        if raw_decision_type == "use_knowledge_base":
            raw_action = "rag_fast_path"
        elif raw_decision_type in {"use_agent", "use_tool"}:
            raw_action = "agent"
        elif raw_decision_type in {"clarify", "direct_answer"}:
            raw_action = raw_decision_type

    # 只在解析旧 Router 输出时接受具体工具 action；生产 Prompt 不再生成它们。
    if raw_action == "search_knowledge_base":
        raw_action = "rag_fast_path"
    elif raw_action in ROUTER_ALLOWED_TOOLS:
        raw_action = "agent"

    if raw_action not in ROUTER_ALLOWED_ACTIONS:
        raise ValueError(f"未知路由 action: {raw_action}")

    decision_type = ROUTER_ACTION_TO_DECISION[raw_action]
    reason = normalize_router_reason(raw_decision)
    confidence = normalize_router_confidence(raw_decision.get("confidence"))
    if raw_action == "rag_fast_path":
        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "rag_fast_path",
                "tool_name": "search_knowledge_base",
                "tool_args": {
                    "query": q,
                    "top_k": top_k,
                },
                "reason": reason,
                "clarification_question": "",
            },
            source="llm",
            confidence=confidence
        )

    if raw_action == "clarify":
        clarification_question = str(
            raw_decision.get("clarification_question")
            or DEFAULT_CLARIFICATION_QUESTION
        ).strip()

        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "clarify",
                "tool_name": "",
                "tool_args": {},
                "reason": reason,
                "clarification_question": clarification_question,
            },
            source="llm",
            confidence=confidence
        )

    if raw_action == "direct_answer":
        return add_router_metadata(
            {
                "decision": decision_type,
                "action": "direct_answer",
                "tool_name": "",
                "tool_args": {},
                "reason": reason,
                "clarification_question": "",
            },
            source="llm",
            confidence=confidence
        )

    return add_router_metadata(
        {
            "decision": decision_type,
            "action": "agent",
            "tool_name": "",
            "tool_args": {},
            "reason": reason,
            "clarification_question": "",
        },
        source="llm",
        confidence=confidence
    )


def route_agent_action_rule_based(question: str, top_k:int) -> dict:
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


def route_agent_action(question: str, top_k:int, trace_id: str = "") -> dict:
    """执行不依赖具体工具名称或用户口吻的安全入口分流。"""
    q = normalize_question(question)
    rule_decision = route_agent_action_rule_based(question, top_k)

    return add_router_metadata(
        rule_decision,
        source="rule_based_guardrail" if not q else "rule_based",
        confidence=1.0,
        llm_used=False,
    )
