"""Agent 计划构造模块。

把经过校验的路由决策转换为可由执行器处理的标准计划步骤。
"""

from typing import Any

from app.agent.state import AgentPlanStep
from app.tools.registry import tool_registry


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
    """构建 RAG/短路计划，并兼容旧版具体工具决策。

    正常 ``agent`` 路径由迭代式 Agent Model 直接选择工具，不经过本函数。
    """
    decision = _safe_decision(decision)
    question = (question or "").strip()
    action = str(decision.get("action") or "").strip()

    if action in {"rag_fast_path", "search_knowledge_base"}:
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

    if action == "agent":
        # Graph 正常不会把该空计划交给 Executor；返回空列表可防止 Planner
        # 在未来新增普通工具时重新承担具体工具选择职责。
        return []

    # 旧调用方可能仍传入具体工具 action。这里通过 Registry 做通用兼容，
    # 不再为每个工具维护 Planner 分支。
    legacy_tool_name = str(decision.get("tool_name") or action).strip()
    if (
        decision.get("decision") == "use_tool"
        and tool_registry.get(legacy_tool_name) is not None
    ):
        return [
            _tool_call_step(
                step_id=1,
                tool_name=legacy_tool_name,
                tool_args=_safe_tool_args(decision),
                reason="兼容旧路由决策并调用已注册工具",
            ),
            _final_answer_step(
                step_id=2,
                reason="根据工具结果生成最终回答",
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
