"""迭代式 Agent 使用的可注入 Chat Model 入口。"""

from typing import cast

from langchain_core.language_models.chat_models import BaseChatModel

from app.llm.factory import get_answer_model
from app.tools.providers import ToolProvider, get_default_tool_provider


def get_agent_chat_model() -> BaseChatModel | None:
    """复用现有 OpenAI-compatible 回答模型配置供 Agent Tool Calling。"""
    return cast(BaseChatModel | None, get_answer_model())


def get_agent_tool_provider() -> ToolProvider:
    """返回 Agent 请求使用的 Provider；测试可替换为独立 Registry。"""
    return get_default_tool_provider()


__all__ = ["get_agent_chat_model", "get_agent_tool_provider"]
