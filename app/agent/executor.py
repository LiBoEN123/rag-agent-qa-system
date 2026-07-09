import time
from typing import Any

from app.agent.state import AgentObservation, AgentPlanStep, AgentState
from app.tools import dispatch_tool


def summarize_tool_result(tool_name: str, tool_result: dict) -> str:
    """为工具结果生成简短摘要。"""
    success = bool(tool_result.get("ok", tool_result.get("status") == "success"))
    if not success:
        return "工具执行失败"

    # 工具可以主动返回摘要；没有摘要时，Executor 再按工具名生成兜底描述。
    summary = tool_result.get("summary") or tool_result.get("result_summary")
    if summary:
        return str(summary)

    if tool_name == "search_knowledge_base":
        count = (tool_result.get("output") or {}).get("count", 0)
        return f"检索到 {count} 个候选 chunk"

    if tool_name == "compare_documents":
        return "文档对比工具执行完成"

    if tool_name == "simple_calculator":
        return "计算工具执行完成"

    if tool_name == "list_documents":
        return "文档列表工具执行完成"

    if tool_name == "get_chunk_text":
        return "chunk 原文工具执行完成"

    return f"{tool_name} 工具执行完成"


def build_observation(
    step: AgentPlanStep,
    tool_result: dict,
    cost_ms: int,
) -> AgentObservation:
    """将一次工具调用结果规整为 AgentObservation。"""
    success = bool(tool_result.get("ok", tool_result.get("status") == "success"))
    output = tool_result.get("output") or {}
    error = str(tool_result.get("error") or "")
    if not success and not error:
        error = "Tool execution failed"

    return AgentObservation(
        step_id=step.step_id,
        tool_name=step.tool_name,
        success=success,
        summary=summarize_tool_result(step.tool_name, tool_result),
        output=output,
        error=error,
        cost_ms=cost_ms,
    )


def build_tool_trace_step(
    step: AgentPlanStep,
    tool_result: dict,
    observation: AgentObservation,
) -> dict[str, Any]:
    """构造兼容现有响应结构的单步工具 trace。"""
    trace: dict[str, Any] = {
        "step": "tool_call",
        "step_id": step.step_id,
        "action": step.action,
        "tool_name": step.tool_name,
        "tool_args": step.tool_args,
        "tool_status": tool_result.get("status", ""),
        "success": observation.success,
        "summary": observation.summary,
        "error": observation.error,
        "cost_ms": observation.cost_ms,
    }

    if step.tool_name == "search_knowledge_base":
        # 检索工具的 trace 需要暴露命中数量、分数和简短 hits，方便调试召回质量。
        output = tool_result.get("output") or {}
        trace.update(
            {
                "final_chunk_count": output.get("count", 0),
                "top_chunk_ids": output.get("top_chunk_ids", []),
                "top_scores": output.get("top_scores", []),
            }
        )

        hits = []
        for chunk in output.get("chunks", [])[:10]:
            # chunk 结构可能来自不同检索阶段，字段缺失时保持 trace 可生成。
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            hits.append(
                {
                    "chunk_id": chunk.get("chunk_id") or metadata.get("chunk_id"),
                    "rerank": chunk.get("rerank_score"),
                    "score": chunk.get("hybrid_score"),
                    "bm25": chunk.get("bm25_score"),
                    "vector": chunk.get("vector_score"),
                    "file_name": chunk.get("file_name") or metadata.get("file_name"),
                }
            )
        trace["hits"] = hits
    else:
        # 普通工具输出体量较小，保留原始结果供评测和故障排查使用。
        trace["tool_output"] = observation.output

    return trace


def _call_dispatch_tool(step: AgentPlanStep, trace_id: str = "") -> dict:
    """调用工具分发入口并传递当前 trace_id。"""
    return dispatch_tool(step.tool_name, step.tool_args, trace_id=trace_id)


def _execute_tool_step(state: AgentState, step: AgentPlanStep) -> bool:
    """执行单个 tool_call 步骤，返回是否可以继续执行后续计划。"""
    step.status = "running"
    start = time.perf_counter()

    try:
        tool_result = _call_dispatch_tool(step, trace_id=state.trace_id)
    except Exception as exc:
        cost_ms = int((time.perf_counter() - start) * 1000)
        tool_result = {
            "status": "error",
            "ok": False,
            "tool_name": step.tool_name,
            "tool_input": step.tool_args,
            "output": {},
            "error": str(exc),
        }
    else:
        cost_ms = int((time.perf_counter() - start) * 1000)

    # Executor 只接受 dict 形态的工具结果；异常形态统一转成工具失败观察。
    if not isinstance(tool_result, dict):
        tool_result = {
            "status": "error",
            "ok": False,
            "tool_name": step.tool_name,
            "tool_input": step.tool_args,
            "output": {},
            "error": f"工具返回值类型异常：{type(tool_result)}",
        }

    observation = build_observation(step, tool_result, cost_ms)
    state.observations.append(observation)
    state.tool_trace.append(build_tool_trace_step(step, tool_result, observation))

    # 工具失败会中止整个执行计划，避免后续步骤基于无效观察继续推进。
    if observation.success:
        step.status = "success"
        return True

    step.status = "failed"
    state.status = "tool_error"
    state.reason = observation.error or "Tool execution failed"
    return False


def execute_plan(state: AgentState) -> AgentState:
    """按 AgentState.plan 顺序执行计划，并直接更新传入的 state。"""
    if not state.plan:
        state.status = "error"
        state.reason = "Agent plan is empty"
        state.tool_trace.append(
            {
                "step": "executor",
                "status": "error",
                "reason": state.reason,
            }
        )
        return state

    for index, step in enumerate(state.plan):
        # 最多处理 max_steps 个步骤；已经短路停止的状态不再覆盖成超限错误。
        if index >= state.max_steps:
            if state.status not in {"clarify", "direct_answer", "tool_error"}:
                state.status = "error"
                state.reason = "Agent exceeded max_steps"
                state.tool_trace.append(
                    {
                        "step": "executor",
                        "status": "error",
                        "reason": state.reason,
                        "max_steps": state.max_steps,
                        "plan_size": len(state.plan),
                    }
                )
            return state

        if step.action == "tool_call":
            ok = _execute_tool_step(state, step)
            if not ok:
                return state
            continue

        # 澄清和直接回答都属于计划短路，不进入工具调用链。
        if step.action == "clarify":
            step.status = "success"
            state.status = "clarify"
            state.reason = step.reason
            state.final_answer = step.reason
            state.tool_trace.append(
                {
                    "step": "clarify",
                    "step_id": step.step_id,
                    "status": "success",
                    "reason": step.reason,
                }
            )
            return state

        # direct_answer 先临时把 reason 作为 final_answer，后续 answer 模块可覆盖。
        if step.action == "direct_answer":
            step.status = "success"
            state.status = "direct_answer"
            state.reason = step.reason
            state.final_answer = step.reason
            state.tool_trace.append(
                {
                    "step": "direct_answer",
                    "step_id": step.step_id,
                    "status": "success",
                    "reason": step.reason,
                }
            )
            return state

        # Executor 只标记 final_answer 已到达，不在这里生成自然语言最终答案。
        if step.action == "final_answer":
            step.status = "skipped"
            state.tool_trace.append(
                {
                    "step": "final_answer",
                    "step_id": step.step_id,
                    "status": "skipped",
                    "reason": "final answer will be composed later",
                }
            )
            if state.status == "running":
                state.status = "ok"
                state.reason = "Agent plan executed"
            continue

        # 未知 action 说明 Planner 与 Executor 的动作协议不一致，直接报错停止。
        step.status = "failed"
        state.status = "error"
        state.reason = f"Unknown agent step action: {step.action}"
        state.tool_trace.append(
            {
                "step": "unknown_action",
                "step_id": step.step_id,
                "action": step.action,
                "status": "failed",
                "reason": state.reason,
            }
        )
        return state

    if state.status == "running":
        state.status = "ok"
        state.reason = "Agent plan executed"

    return state
