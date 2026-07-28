"""PR3 状态字段和 Guardrail ToolMessage 的 SQLite Checkpointer 回归。"""

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from app.agent import framework_agent, framework_tools
from app.graph import builder as graph_builder
from app.graph import checkpointing, nodes as graph_nodes
from app.graph.runtime import run_agent_langgraph
from app.tools import runtime as tool_runtime
from app.tools.contracts import ToolSpec
from app.tools.providers import LocalToolProvider
from app.tools.registry import ToolRegistry
from app.tools.result import tool_success


class CheckpointArgs(BaseModel):
    value: int


class SequenceModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


def _call(value: int | None, call_id: str) -> dict:
    return {
        "name": "checkpoint_tool",
        "args": {} if value is None else {"value": value},
        "id": call_id,
        "type": "tool_call",
    }


def _assert_no_dangling_tool_calls(messages) -> None:
    requested_ids = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    returned_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    assert returned_ids == requested_ids


def test_sqlite_serializes_success_correction_repeat_and_budget_stops(
    monkeypatch,
    tmp_path,
):
    registry = ToolRegistry()
    handler_calls: list[int] = []

    def handler(args, context):
        handler_calls.append(args["value"])
        return tool_success(
            "checkpoint_tool",
            args,
            {"value": args["value"]},
        )

    registry.register(
        ToolSpec(
            name="checkpoint_tool",
            version="1.0.0",
            description="验证 Checkpointer 序列化的受控工具",
            args_schema={"value": "integer"},
            output_schema={"value": "integer"},
            handler=handler,
            args_model=CheckpointArgs,
        )
    )
    monkeypatch.setattr(
        framework_tools,
        "get_agent_tool_provider",
        lambda: LocalToolProvider(registry),
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_ITERATIONS", 10)
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOOL_ERRORS", 3)
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS",
        2,
    )
    monkeypatch.setattr(
        graph_nodes.Settings,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS",
        2,
    )
    monkeypatch.setattr(graph_nodes.Settings, "AGENT_MAX_TOTAL_COST_MS", 120000)
    monkeypatch.setattr(graph_nodes.Settings, "TOOL_AUDIT_ENABLED", False)
    monkeypatch.setattr(
        checkpointing.Settings,
        "LANGGRAPH_CHECKPOINTER",
        "sqlite",
    )
    monkeypatch.setattr(
        checkpointing.Settings,
        "LANGGRAPH_CHECKPOINT_DB_PATH",
        tmp_path / "guardrails.sqlite3",
    )
    cases = [
        (
            "success",
            SequenceModel(
                [
                    AIMessage(content="", tool_calls=[_call(1, "success-call")]),
                    AIMessage(content="success"),
                ]
            ),
            10,
            "OK",
            "completed",
        ),
        (
            "correction",
            SequenceModel(
                [
                    AIMessage(content="", tool_calls=[_call(None, "invalid-call")]),
                    AIMessage(content="", tool_calls=[_call(2, "corrected-call")]),
                    AIMessage(content="corrected"),
                ]
            ),
            10,
            "OK",
            "completed",
        ),
        (
            "repeat",
            SequenceModel(
                [
                    AIMessage(content="", tool_calls=[_call(3, "repeat-1")]),
                    AIMessage(content="", tool_calls=[_call(3, "repeat-2")]),
                    AIMessage(content="", tool_calls=[_call(3, "repeat-3")]),
                ]
            ),
            10,
            "MODEL_ERROR",
            "repeated_tool_call",
        ),
        (
            "budget",
            SequenceModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _call(4, "budget-1"),
                            _call(5, "budget-2"),
                        ],
                    )
                ]
            ),
            1,
            "MODEL_ERROR",
            "max_tool_calls_reached",
        ),
    ]

    graph_builder.reset_agent_graph_runtime()
    try:
        for label, model, max_tool_calls, expected_status, expected_reason in cases:
            monkeypatch.setattr(
                framework_agent,
                "get_agent_chat_model",
                lambda selected=model: selected,
            )
            monkeypatch.setattr(
                graph_nodes.Settings,
                "AGENT_MAX_TOOL_CALLS",
                max_tool_calls,
            )
            graph_builder.get_compiled_agent_graph.cache_clear()
            result = run_agent_langgraph(
                write_bad_cases=lambda record: None,
                original_question=f"checkpoint {label}",
                effective_question=f"checkpoint {label}",
                top_k=5,
                decompose_top_k=10,
                trace_id=f"trace-checkpoint-{label}",
                session_id="session-checkpoint",
            )
            thread_id = checkpointing.build_checkpoint_thread_id(
                trace_id=f"trace-checkpoint-{label}",
                session_id="session-checkpoint",
            )
            saved = checkpointing.get_checkpointer().get_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            values = saved.checkpoint["channel_values"]

            assert result["status"] == expected_status
            assert values["stop_reason"] == expected_reason
            assert isinstance(values["agent_elapsed_ms"], int)
            assert isinstance(values["last_tool_call_fingerprint"], str)
            assert isinstance(values["last_tool_error_fingerprint"], str)
            _assert_no_dangling_tool_calls(values["messages"])
    finally:
        graph_builder.reset_agent_graph_runtime()
        tool_runtime.reset_tool_runtime()

    # success=1、correction=1、repeat=2、budget=1（框架允许预算内调用）。
    assert handler_calls == [1, 2, 3, 3, 4]
