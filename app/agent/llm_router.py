"""Transport and parsing for the optional LLM Agent router."""

import json
import time

import requests

from app.common.logging import get_logger
from app.config import Settings
from app.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from app.tools.registry import tool_registry


logger = get_logger("chat_service")


def build_router_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if Settings.ROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.ROUTER_API_KEY}"
    return headers


def parse_router_json_output(llm_output: str) -> dict:
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
    allowed_actions = set(tool_registry.names()) | {"clarify", "direct_answer"}
    payload = {
        "model": Settings.ROUTER_MODEL_NAME,
        "temperature": Settings.ROUTER_TEMPERATURE,
        "max_tokens": Settings.ROUTER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ROUTER_USER_PROMPT.format(
                    question=question,
                    top_k=top_k,
                    available_actions=json.dumps(
                        sorted(allowed_actions), ensure_ascii=False
                    ),
                    tool_specs=json.dumps(
                        tool_registry.router_specs(), ensure_ascii=False, indent=2
                    ),
                ),
            },
        ],
    }
    total_attempts = Settings.ROUTER_MAX_RETRIES + 1
    last_error = ""
    for attempt in range(total_attempts):
        try:
            response = requests.post(
                Settings.ROUTER_API_URL,
                headers=build_router_headers(),
                json=payload,
                timeout=Settings.ROUTER_TIMEOUT,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = str(exc)
            if attempt < Settings.ROUTER_MAX_RETRIES:
                logger.warning(
                    "[trace_id=%s] [router_llm] retry attempt=%s/%s error=%s",
                    trace_id,
                    attempt + 1,
                    total_attempts,
                    last_error,
                )
                time.sleep(Settings.LLM_RETRY_SLEEP)
    raise RuntimeError(f"路由模型调用失败：{last_error}")
