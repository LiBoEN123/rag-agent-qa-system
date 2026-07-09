"""Agent 公共入口。

对外暴露路由能力，并负责把可替换的 LLM 路由依赖同步到具体实现。
"""

from app.agent import router as _router
from app.agent.router import *  # noqa: F401,F403


def route_agent_action(
    question: str,
    top_k: int,
    trace_id: str = "",
) -> dict:
    """执行 Agent 路由，并同步门面层可能被替换的 LLM 路由函数。"""
    # Preserve monkeypatching app.agent.route_agent_action_by_llm.
    _router.route_agent_action_by_llm = route_agent_action_by_llm
    return _router.route_agent_action(question, top_k, trace_id=trace_id)
