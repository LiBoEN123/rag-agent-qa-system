"""LLM 路由调用模块。

负责构造路由模型请求，并解析模型返回的 JSON 决策。
"""

import json

from app.llm.factory import get_router_model
from app.llm.gateway import ModelGateway
from app.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT


def parse_router_json_output(llm_output: str) -> dict:
    """解析路由模型输出，并兼容 Markdown JSON 代码块。"""
    output = (llm_output or "").strip()
    if output.startswith("```"):
        output = output.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("路由模型未返回 JSON 对象")
        data = json.loads(output[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("路由模型返回值不是 JSON 对象")
    return data


def call_router_llm(question: str, top_k: int, trace_id: str = "") -> str:
    """通过统一 Gateway 调用只做入口判断的 LLM Router。"""
    allowed_actions = {
        "agent",
        "rag_fast_path",
        "clarify",
        "direct_answer",
    }
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ROUTER_USER_PROMPT.format(
                question=question,
                top_k=top_k,
                available_actions=json.dumps(
                    sorted(allowed_actions), ensure_ascii=False
                ),
            ),
        },
    ]
    gateway = ModelGateway(
        role="router",
        langchain_model=get_router_model(),
    )
    return gateway.invoke_text(messages, trace_id=trace_id)
