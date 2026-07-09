"""Agent 计划构造模块。

把经过校验的路由决策转换为可由执行器处理的标准计划步骤。
"""

from typing import Any

from app.agent.state import AgentPlanStep


def _safe_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
    """确保路由决策是可安全读取的字典。"""
    return decision if isinstance(decision, dict) else {}


def _safe_tool_args(decision: dict[str, Any]) -> dict[str, Any]:
    """从路由决策中提取工具参数，格式异常时返回空字典。"""
    tool_args = decision.get("tool_args")
    return tool_args if isinstance(tool_args, dict) else {}


def _tool_call_step(
    step_id: int,
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str,
) -> AgentPlanStep:
    """构造一个工具调用类型的计划步骤。"""
    return AgentPlanStep(
        step_id=step_id,
        action="tool_call",
        tool_name=tool_name,
        tool_args=tool_args,
        reason=reason,
    )


def _final_answer_step(step_id: int, reason: str) -> AgentPlanStep:
    """构造一个生成最终回答类型的计划步骤。"""
    return AgentPlanStep(
        step_id=step_id,
        action="final_answer",
        reason=reason,
    )


def _fallback_rag_plan(
    question: str,
    top_k: int,
    reason: str = "未知 action，回退为知识库检索",
) -> list[AgentPlanStep]:
    """在无法识别路由 action 时构造默认的 RAG 检索计划。"""
    return [
        _tool_call_step(
            step_id=1,
            tool_name="search_knowledge_base",
            tool_args={"query": question, "top_k": top_k},
            reason=reason,
        ),
        _final_answer_step(
            step_id=2,
            reason="根据检索证据生成最终回答",
        ),
    ]


def build_plan_from_decision(
    decision: dict[str, Any] | None,
    question: str,
    top_k: int = 5,
) -> list[AgentPlanStep]:
    """根据路由决策构建 Agent v2 的执行计划。"""
    decision = _safe_decision(decision)
    question = (question or "").strip()
    action = str(decision.get("action") or "").strip()

    if action == "search_knowledge_base":
        tool_args = _safe_tool_args(decision)
        if not tool_args:
            tool_args = {"query": question, "top_k": top_k}

        return [
            _tool_call_step(
                step_id=1,
                tool_name="search_knowledge_base",
                tool_args=tool_args,
                reason="先检索知识库证据",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据检索证据生成最终回答",
            ),
        ]

    if action == "compare_documents":
        return [
            _tool_call_step(
                step_id=1,
                tool_name="compare_documents",
                tool_args=_safe_tool_args(decision),
                reason="先调用文档对比工具获取差异证据",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据文档对比结果生成最终回答",
            ),
        ]

    if action == "simple_calculator":
        return [
            _tool_call_step(
                step_id=1,
                tool_name="simple_calculator",
                tool_args=_safe_tool_args(decision),
                reason="先调用计算器工具得到计算结果",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据计算结果生成最终回答",
            ),
        ]

    if action == "list_documents":
        return [
            _tool_call_step(
                step_id=1,
                tool_name="list_documents",
                tool_args=_safe_tool_args(decision),
                reason="先列出知识库文档",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据文档列表生成最终回答",
            ),
        ]

    if action == "get_chunk_text":
        return [
            _tool_call_step(
                step_id=1,
                tool_name="get_chunk_text",
                tool_args=_safe_tool_args(decision),
                reason="先查看指定 chunk 原文",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据 chunk 原文生成最终回答",
            ),
        ]

    if action == "clarify":
        reason = (
            str(decision.get("clarification_question") or "").strip()
            or str(decision.get("reason") or "").strip()
            or "问题需要进一步澄清"
        )
        return [
            AgentPlanStep(
                step_id=1,
                action="clarify",
                reason=reason,
            )
        ]

    if action == "direct_answer":
        return [
            AgentPlanStep(
                step_id=1,
                action="direct_answer",
                reason=str(decision.get("reason") or "该问题可以直接回答"),
            )
        ]

    return _fallback_rag_plan(question, top_k)


__all__ = ["build_plan_from_decision"]
