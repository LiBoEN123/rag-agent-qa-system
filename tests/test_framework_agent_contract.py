"""LangChain/LangGraph Agent 子图的关键版本契约。"""

from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from app.agent.framework_agent import build_framework_agent


class _ProbeModel:
    profile: dict = {}

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def invoke(self, messages):
        del messages
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "handoff_probe",
                    "args": {"value": 7},
                    "id": "handoff-probe-call",
                    "type": "tool_call",
                }
            ],
        )


class _ParentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    handoff_value: int
    reached_rag: bool


def test_production_agent_is_framework_model_tool_graph():
    """生产子图实际包含 create_agent 的标准 model/tools 节点与调用限制。"""
    node_names = set(build_framework_agent().get_graph().nodes)

    assert {"model", "tools"} <= node_names
    assert "ModelCallLimitMiddleware.before_model" in node_names
    assert "ToolCallLimitMiddleware.after_model" in node_names


def test_create_agent_tool_command_handoff_to_parent():
    """工具的 Command.PARENT 可携带配对消息并跳入父图 RAG 节点。"""

    @tool
    def handoff_probe(value: int, runtime: ToolRuntime) -> Command:
        """验证 Agent 子图到父图的公开 Command 交接。"""
        ai_message = runtime.state["messages"][-1]
        return Command(
            graph=Command.PARENT,
            goto="rag",
            update={
                "messages": [
                    ai_message,
                    ToolMessage(
                        content="handoff",
                        tool_call_id=runtime.tool_call_id,
                    ),
                ],
                "handoff_value": value,
            },
        )

    agent = create_agent(
        _ProbeModel(),
        [handoff_probe],
        name="handoff_contract_agent",
    )
    builder = StateGraph(_ParentState)
    builder.add_node("agent", agent)
    builder.add_node("rag", lambda state: {"reached_rag": True})
    builder.add_edge(START, "agent")
    builder.add_edge("rag", END)

    output = builder.compile().invoke(
        {"messages": [HumanMessage(content="handoff")]}
    )

    assert output["handoff_value"] == 7
    assert output["reached_rag"] is True
    assert isinstance(output["messages"][-2], AIMessage)
    assert output["messages"][-2].tool_calls[0]["id"] == "handoff-probe-call"
    assert isinstance(output["messages"][-1], ToolMessage)
    assert output["messages"][-1].tool_call_id == "handoff-probe-call"
