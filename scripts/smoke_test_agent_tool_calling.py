"""真实模型供应商 Tool Calling 协议的手动冒烟程序。

用途：绑定当前工具 Schema，确认供应商返回工具名、参数和 tool_call_id 的协议形态。
输入：当前回答模型配置、工具注册合同，以及程序内固定的文档列表和知识库检索场景。
输出：脱敏后的 Schema/tool_calls 诊断信息；协议不符合预期时以异常失败。
边界：会调用真实模型，但不会执行模型返回的工具；不会由 Pytest 或 CI 自动运行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.framework_tools import get_local_tools  # noqa: E402
from app.llm.factory import get_answer_model  # noqa: E402
from app.tools.context import build_default_tool_context  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.common.redaction import redact_sensitive_data  # noqa: E402


SCENARIOS = (
    (
        "list_documents",
        "请列出知识库中的文档，必须调用文档列表工具。",
    ),
    (
        "search_knowledge_base",
        "请从知识库检索奖学金申请条件，必须调用知识库检索工具。",
    ),
)


def _print_json(label: str, value: object) -> None:
    """脱敏并稳定序列化一项协议诊断信息。"""
    print(
        f"{label}: "
        + json.dumps(
            redact_sensitive_data(value),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def main() -> int:
    """绑定真实工具 Schema 并验证两类真实模型 tool_calls。"""
    model = get_answer_model()
    if model is None:
        raise RuntimeError("回答模型不可用，请先配置 LLM_API_KEY 和模型地址")

    context = build_default_tool_context(source="manual_tool_calling_smoke")
    from langchain.agents.middleware import ModelRequest
    from langgraph.runtime import Runtime
    from app.agent.tool_visibility import ProjectToolVisibilityMiddleware

    tools = ProjectToolVisibilityMiddleware.visible_request(ModelRequest(
        model=model, messages=[], tools=list(get_local_tools(get_settings())),
        runtime=Runtime(context=context),
    )).tools
    tool_map = {tool.name: tool for tool in tools}
    print("bound_tool_names:", sorted(tool_map))
    for required_name in ("search_knowledge_base", "list_documents"):
        tool = tool_map.get(required_name)
        if tool is None:
            raise RuntimeError(f"模型工具列表缺少 {required_name}")
        schema = tool.tool_call_schema.model_json_schema()
        _print_json(f"{required_name}.args_schema", schema)

    bound_model = model.bind_tools(tools)
    for expected_tool, prompt in SCENARIOS:
        # 这里只检查供应商返回的 tool_calls，不执行对应的真实工具。
        response = bound_model.invoke([HumanMessage(content=prompt)])
        print("response_type:", type(response).__name__)
        if not isinstance(response, AIMessage):
            raise RuntimeError("供应商没有返回 AIMessage")
        _print_json("response.tool_calls", response.tool_calls)
        _print_json("response.additional_kwargs", response.additional_kwargs)
        if not response.tool_calls:
            raise RuntimeError(f"场景 {expected_tool} 没有返回 tool_calls")
        first_call = response.tool_calls[0]
        print("tool_name:", first_call.get("name", ""))
        _print_json("tool_args", first_call.get("args", {}))
        print("tool_call_id:", first_call.get("id", ""))
        if first_call.get("name") != expected_tool:
            raise RuntimeError(
                f"期望工具 {expected_tool}，实际为 {first_call.get('name')}"
            )
        if not first_call.get("id"):
            raise RuntimeError("供应商 tool_call 缺少 tool_call_id")
    print("真实模型 Tool Calling 冒烟验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
