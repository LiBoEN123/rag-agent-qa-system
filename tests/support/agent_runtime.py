"""直接 ``create_agent`` Runtime 的测试执行辅助。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from app.agent.results import (
    build_framework_agent_result,
    build_short_circuit_result,
)
from app.agent.runtime import prepare_agent_runtime_invocation
from app.config import AppSettings, get_settings
from app.tools.context import ToolInvocationContext


class FakeToolCallingModel:
    """按顺序返回 AIMessage，并记录工具绑定和每轮完整输入。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tool_names: list[str] = []
        self.received_messages: list[list] = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.received_messages.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AgentRuntimeHarness:
    """让治理集成测试取得框架状态和最终公共 Result。"""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def nodes(self):
        """暴露 ``create_agent`` 自身节点，不包含项目 Parent Graph 节点。"""
        return self.get_graph().nodes

    def _prepare(
        self,
        input_value: Mapping[str, Any],
        context: ToolInvocationContext | None,
    ):
        return prepare_agent_runtime_invocation(
            top_k=int(input_value.get("top_k", self.settings.top_k)),
            decompose_top_k=int(
                input_value.get(
                    "decompose_top_k",
                    self.settings.decompose_top_k,
                )
            ),
            trace_id=str(input_value.get("trace_id") or ""),
            session_id=input_value.get("session_id"),
            original_question=str(input_value.get("original_question") or ""),
            effective_question=str(input_value.get("effective_question") or ""),
            tool_context=context,
            settings=self.settings,
        )

    @staticmethod
    def _merge_state(
        raw: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        status_map = {
            "OK": "ok",
            "NEED_CLARIFICATION": "clarify",
            "NO_RELEVANT_INFO": "no_evidence",
            "TOOL_ERROR": "tool_error",
            "MODEL_ERROR": "model_error",
            "INTERNAL_ERROR": "error",
        }
        return {
            **raw,
            "status": status_map.get(str(result.get("status") or ""), "error"),
            "final_answer": str(result.get("answer") or ""),
            "result": dict(result),
        }

    async def astream(
        self,
        input_value: Mapping[str, Any],
        *,
        context: ToolInvocationContext | None = None,
        stream_mode: str = "values",
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """异步执行直接 Agent；values 模式最后追加归一化快照。"""
        invocation = self._prepare(input_value, context)
        if invocation.agent is None:
            yield self._merge_state({}, build_short_circuit_result(invocation))
            return
        raw: Mapping[str, Any] | None = None
        async for part in invocation.agent.astream(
            invocation.agent_input,
            config=invocation.agent_config,
            context=invocation.tool_context,
            stream_mode=stream_mode,
            **kwargs,
        ):
            if stream_mode == "values" and isinstance(part, Mapping):
                raw = part
            yield part
        if stream_mode == "values":
            assert raw is not None
            result = build_framework_agent_result(invocation, raw)
            yield self._merge_state(raw, result)

    def get_graph(self, **kwargs):
        """返回 ``create_agent`` 自身结构供架构断言使用。"""
        invocation = self._prepare(
            {
                "original_question": "测试 Agent Runtime",
                "effective_question": "测试 Agent Runtime",
                "top_k": self.settings.top_k,
                "decompose_top_k": self.settings.decompose_top_k,
            },
            None,
        )
        assert invocation.agent is not None
        return invocation.agent.get_graph(**kwargs)


def build_agent_harness(
    settings: AppSettings | None = None,
) -> AgentRuntimeHarness:
    """构造不含 Parent Graph 的直接 Agent 测试辅助。"""
    return AgentRuntimeHarness(settings)


__all__ = ["AgentRuntimeHarness", "FakeToolCallingModel", "build_agent_harness"]
