"""Public Agent API and backwards-compatible router entry points."""

from app.agent import router as _router
from app.agent.router import *  # noqa: F401,F403


def route_agent_action(
    question: str,
    top_k: int,
    trace_id: str = "",
) -> dict:
    # Preserve monkeypatching app.agent.route_agent_action_by_llm.
    _router.route_agent_action_by_llm = route_agent_action_by_llm
    return _router.route_agent_action(question, top_k, trace_id=trace_id)
