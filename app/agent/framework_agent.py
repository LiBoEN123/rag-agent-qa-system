"""构建项目唯一的 LangChain ``create_agent`` Agent Loop。"""

import operator
from collections.abc import Sequence
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
)
from langchain.agents.middleware.types import AgentState as LangChainAgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from typing_extensions import NotRequired

from app.agent.final_response import FinalAgentResponse
from app.agent.framework_middleware import (
    _MODEL_GOVERNANCE_KEY,
    ProjectToolCallLimitMiddleware,
    ProjectToolGovernanceMiddleware,
)
from app.agent.framework_tools import resolve_agent_tools
from app.agent.tool_visibility import ProjectToolVisibilityMiddleware
from app.config import AppSettings, get_settings
from app.llm.factory import get_answer_model
from app.prompts import AGENT_SYSTEM_PROMPT
from app.tools.context import ToolInvocationContext, build_default_tool_context


def _prefer_stop_reason(left: str, right: str) -> str:
    """归并并优先保留最新的非空停止原因。"""
    return right or left


class FrameworkAgentState(LangChainAgentState):
    """create_agent 内部使用的最小治理状态。"""

    governance_events: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    stop_reason: NotRequired[Annotated[str, _prefer_stop_reason]]


class _DeferredAgentModel:
    """把模型实例解析延迟到请求执行，构图时不触发模型配置。"""

    profile: dict[str, Any] = {}

    @staticmethod
    def _model():
        """延迟取得当前请求使用的 Agent 模型。"""
        model = get_answer_model()
        if model is None:
            raise RuntimeError("Agent model is unavailable")
        return model

    def bind_tools(self, tools, **kwargs):
        """绑定静态请求工具，并返回记录治理事件的模型适配器。"""
        clean_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        bound = self._model().bind_tools(tools, **clean_kwargs)
        return _GovernedBoundModel(
            bound,
            available_tool_names=[
                tool.name for tool in tools if tool.name != FinalAgentResponse.__name__
            ],
        )

    def bind(self, **kwargs):
        """兼容转发模型的通用绑定参数。"""
        model = self._model()
        bind = getattr(model, "bind", None)
        bound = bind(**kwargs) if callable(bind) else model
        return _GovernedBoundModel(bound, available_tool_names=[])


class _GovernedBoundModel:
    """在模型边界记录成功/失败，交给 ``aafter_model`` 写入状态。"""

    # 保存底层模型、可用工具名称及模型能力信息，供调用和治理记录使用。
    def __init__(self, model: Any, *, available_tool_names: list[str]) -> None:
        self._model = model
        self._available_tool_names = list(available_tool_names)
        self.profile = getattr(model, "profile", {})

    # 为模型消息附加成功状态、调用耗时和工具请求统计。
    def _success(self, output: Any, started: int) -> Any:
        if not isinstance(output, AIMessage):
            return output
        business_calls = [
            call for call in output.tool_calls
            if call["name"] != FinalAgentResponse.__name__
        ]
        event = {
            "kind": "model",
            "ok": True,
            "cost_ms": max(0, _now_ms() - started),
            "has_tool_calls": bool(business_calls),
            "requested_tool_call_count": len(business_calls),
            "available_tool_names": self._available_tool_names,
        }
        return output.model_copy(
            update={
                "additional_kwargs": {
                    **output.additional_kwargs,
                    _MODEL_GOVERNANCE_KEY: event,
                }
            }
        )

    # 将模型异常转换为携带失败治理记录的统一模型消息。
    @staticmethod
    def _failure(started: int, exc: Exception) -> AIMessage:
        event = {
            "kind": "model",
            "ok": False,
            "cost_ms": max(0, _now_ms() - started),
            "error_type": type(exc).__name__,
        }
        return AIMessage(
            content="Agent model unavailable",
            additional_kwargs={_MODEL_GOVERNANCE_KEY: event},
        )

    async def ainvoke(self, messages):
        """异步调用实际绑定模型并保持相同事件语义。"""
        started = _now_ms()
        try:
            return self._success(await self._model.ainvoke(messages), started)
        except Exception as exc:
            return self._failure(started, exc)


def build_framework_agent(
    settings: AppSettings | None = None,
    *,
    tools: Sequence[BaseTool] | None = None,
    governance: ProjectToolGovernanceMiddleware | None = None,
):
    """创建使用请求可见工具和内置调用限制的唯一 Agent Loop。"""
    resolved = settings or get_settings()
    resolved_governance = governance or ProjectToolGovernanceMiddleware(
        settings=resolved
    )
    resolved_tools = list(tools) if tools is not None else resolve_agent_tools(
        build_default_tool_context(
            source="agent_runtime",
            settings=resolved,
        ),
        resolved,
    )
    agent = create_agent(
        _DeferredAgentModel(),
        tools=resolved_tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        response_format=ToolStrategy(FinalAgentResponse, handle_errors=True),
        middleware=[
            ProjectToolVisibilityMiddleware(),
            resolved_governance,
            ModelCallLimitMiddleware(
                run_limit=resolved.agent_max_iterations,
                exit_behavior="end",
            ),
            ProjectToolCallLimitMiddleware(
                run_limit=resolved.agent_max_tool_calls,
                exit_behavior="continue",
            ),
        ],
        state_schema=FrameworkAgentState,
        context_schema=ToolInvocationContext,
        name="framework_agent",
    )
    # ToolNode 对同批调用使用 Send；顺序执行使 terminal fail-closed 可确定。
    return agent.with_config({"max_concurrency": 1})


def _now_ms() -> int:
    """复用治理 Middleware 的可测试时钟。"""
    from app.agent.framework_middleware import _now_ms as middleware_now_ms

    return middleware_now_ms()


__all__ = [
    "FrameworkAgentState",
    "build_framework_agent",
]
