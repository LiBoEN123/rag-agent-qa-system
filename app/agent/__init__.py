"""Agent 公共入口。

对外暴露路由能力，并负责把可替换的 LLM 路由依赖同步到具体实现。
"""

from app.agent import router as _router
from app.agent.formatting import format_tool_answer
from app.agent.llm_router import (
    build_router_headers,
    call_router_llm,
    parse_router_json_output,
)
from app.agent.router import (
    DEFAULT_CLARIFICATION_QUESTION,
    HIGH_CONFIDENCE_RULE_ACTIONS,
    ROUTER_ACTION_TO_DECISION,
    ROUTER_ALLOWED_ACTIONS,
    ROUTER_ALLOWED_TOOLS,
    ROUTER_TOOL_SPECS,
    add_router_metadata,
    normalize_chunk_id_value,
    normalize_router_confidence,
    normalize_router_reason,
    route_agent_action_by_llm,
    route_agent_action_rule_based,
    validate_route_decision,
)
from app.agent.rules import (
    COMPARE_DOCUMENT_KEYWORDS,
    build_compare_documents_rule_args,
    extract_chunk_id,
    extract_simple_math_expression,
    is_safe_math_expression,
    normalize_math_text,
    normalize_question,
)
from app.config import Settings


__all__ = [
    "COMPARE_DOCUMENT_KEYWORDS",
    "DEFAULT_CLARIFICATION_QUESTION",
    "HIGH_CONFIDENCE_RULE_ACTIONS",
    "ROUTER_ACTION_TO_DECISION",
    "ROUTER_ALLOWED_ACTIONS",
    "ROUTER_ALLOWED_TOOLS",
    "ROUTER_TOOL_SPECS",
    "Settings",
    "add_router_metadata",
    "build_compare_documents_rule_args",
    "build_router_headers",
    "call_router_llm",
    "extract_chunk_id",
    "extract_simple_math_expression",
    "format_tool_answer",
    "is_safe_math_expression",
    "normalize_chunk_id_value",
    "normalize_math_text",
    "normalize_question",
    "normalize_router_confidence",
    "normalize_router_reason",
    "parse_router_json_output",
    "route_agent_action",
    "route_agent_action_by_llm",
    "route_agent_action_rule_based",
    "validate_route_decision",
]


def route_agent_action(
    question: str,
    top_k: int,
    trace_id: str = "",
) -> dict:
    """执行 Agent 路由，并同步门面层可能被替换的 LLM 路由函数。"""
    # Preserve monkeypatching app.agent.route_agent_action_by_llm.
    _router.route_agent_action_by_llm = route_agent_action_by_llm
    return _router.route_agent_action(question, top_k, trace_id=trace_id)
