"""构建项目唯一的 LangChain ``create_agent`` Agent Loop。"""

import operator
from collections.abc import Sequence
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import AgentState as LangChainAgentState
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from typing_extensions import NotRequired

from app.agent.framework_middleware import (
    _MODEL_GOVERNANCE_KEY,
    ProjectToolExecutionMiddleware,
    ProjectToolGovernanceMiddleware,
)
from app.agent.framework_tools import ProjectStructuredTool, resolve_agent_tools
from app.agent.tool_visibility import ProjectToolVisibilityMiddleware
from app.config import AppSettings, get_settings
from app.llm.factory import get_answer_model
from app.tools.context import ToolInvocationContext, build_default_tool_context
from app.tools.execution import project_failure_content, retry_project_failure


def _prefer_stop_reason(left: str, right: str) -> str:
    """归并并优先保留最新的非空停止原因。"""
    return right or left


def _prefer_knowledge_base_result(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """同批知识库调用完成时保留最后一个非空交接结果。"""
    return right or left


class FrameworkAgentState(LangChainAgentState):
    """create_agent 内部使用的最小治理状态。"""

    governance_events: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    agent_elapsed_ms: NotRequired[int]
    stop_reason: NotRequired[Annotated[str, _prefer_stop_reason]]
    knowledge_base_result: NotRequired[
        Annotated[
            dict[str, Any] | None,
            _prefer_knowledge_base_result,
        ]
    ]


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
            available_tool_names=[tool.name for tool in tools],
        )

    def bind(self, **kwargs):
        """兼容转发模型的通用绑定参数。"""
        model = self._model()
        bind = getattr(model, "bind", None)
        bound = bind(**kwargs) if callable(bind) else model
        return _GovernedBoundModel(bound, available_tool_names=[])


class _GovernedBoundModel:
    """在模型边界记录成功/失败，交给 ``after_model`` 写入状态。"""

    # 保存底层模型、可用工具名称及模型能力信息，供调用和治理记录使用。
    def __init__(self, model: Any, *, available_tool_names: list[str]) -> None:
        self._model = model
        self._available_tool_names = list(available_tool_names)
        self.profile = getattr(model, "profile", {})

    # 为模型消息附加成功状态、调用耗时和工具请求统计。
    def _success(self, output: Any, started: int) -> Any:
        if not isinstance(output, AIMessage):
            return output
        event = {
            "kind": "model",
            "ok": True,
            "cost_ms": max(0, _now_ms() - started),
            "has_tool_calls": bool(output.tool_calls),
            "requested_tool_call_count": len(output.tool_calls),
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

    def invoke(self, messages):
        """同步调用实际绑定模型并转换异常。"""
        started = _now_ms()
        try:
            return self._success(self._model.invoke(messages), started)
        except Exception as exc:
            return self._failure(started, exc)

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
    local_tools = [
        tool for tool in resolved_tools if isinstance(tool, ProjectStructuredTool)
    ]
    retry_groups: dict[int, list[str]] = {}
    for tool in local_tools:
        spec = tool.project_spec
        if spec.idempotent and spec.max_retries > 0:
            retry_groups.setdefault(spec.max_retries, []).append(tool.name)
    execution_middleware = [
        ProjectToolExecutionMiddleware(),
        ToolErrorMiddleware(
            project_failure_content,
            tools=local_tools,
        ),
        *(
            ToolRetryMiddleware(
                max_retries=max_retries,
                tools=tool_names,
                retry_on=retry_project_failure,
                on_failure="error",
                backoff_factor=2.0,
                initial_delay=resolved.tool_retry_base_delay_ms / 1000.0,
                max_delay=resolved.tool_retry_max_delay_ms / 1000.0,
                jitter=resolved.tool_retry_jitter_ratio > 0,
            )
            for max_retries, tool_names in sorted(retry_groups.items())
        ),
    ] if local_tools else []
    agent = create_agent(
        _DeferredAgentModel(),
        tools=resolved_tools,
        middleware=[
            ProjectToolVisibilityMiddleware(),
            resolved_governance,
            *execution_middleware,
            ModelCallLimitMiddleware(
                run_limit=resolved.agent_max_iterations,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
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
