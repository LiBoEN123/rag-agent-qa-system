"""LangGraph 主图的条件边函数。"""

from typing import Literal

from app.graph.state import AgentGraphState


def route_condition(state: AgentGraphState) -> Literal["route", "initialize"]:
    """仅在调用方未提供路由决策时进入 Router 节点。"""
    return "route" if state.get("decision") is None else "initialize"


__all__ = ["route_condition"]
