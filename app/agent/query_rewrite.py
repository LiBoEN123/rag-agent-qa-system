import json
import re

import requests

from app.config import Settings
from app.common.logging import get_logger


logger = get_logger("chat_service")

OBVIOUS_FOLLOW_UP_MARKERS = [
    "那",
    "这个",
    "那个",
    "它",
    "上面",
    "前面",
    "刚才",
    "继续",
    "还有",
    "遗漏",
    "完整",
    "补充",
    "就这些",
    "都包括",
    "条件呢",
    "材料呢",
    "时间呢",
    "入口呢",
]

SYSTEM_QUESTIONS = {
    "你是谁",
    "你是谁?",
    "你能做什么",
    "你能做什么?",
    "你可以做什么",
    "你可以做什么?",
    "你有什么功能",
    "你有什么功能?",
}
def _normalize_question(question: str) -> str:
    """清理问题文本并统一常见问号。"""
    return (question or "").strip().replace("？", "?")


def _has_usable_history(history: list[dict]) -> bool:
    """判断历史中是否存在可用于改写的消息。"""
    for message in history or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") in {"user", "assistant"} and str(message.get("content") or "").strip():
            return True
    return False


def _is_simple_math_question(question: str) -> bool:
    """判断当前输入是否只是简单四则运算。"""
    compact = re.sub(r"\s+", "", _normalize_question(question))
    compact = compact.rstrip("=?。")

    if not compact:
        return False

    return bool(
        re.fullmatch(r"[0-9+\-*/().]+", compact)
        and re.search(r"[+\-*/]", compact)
    )


def _is_system_capability_question(question: str) -> bool:
    """判断是否为系统能力类问题。"""
    compact = _normalize_question(question).rstrip("。")
    return compact in SYSTEM_QUESTIONS


def _last_user_question(history: list[dict]) -> str:
    """从历史中取最近一条用户问题。"""
    for message in reversed(history or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return ""


def _fallback_rewrite(question: str, history: list[dict]) -> str:
    """在 LLM 不可用时用上一轮问题拼接追问。"""
    previous_question = _last_user_question(history)
    if not previous_question:
        return question
    return f"基于“{previous_question}”，追问“{question}”"


def _has_obvious_follow_up_marker(question: str) -> bool:
    """判断问题是否包含明显追问标记。"""
    q = _normalize_question(question)
    return any(marker in q for marker in OBVIOUS_FOLLOW_UP_MARKERS)


def should_check_rewrite_with_history(
    question: str,
    history: list[dict],
) -> tuple[bool, str]:
    """判断是否需要让 LLM 检查问题对历史的依赖，不直接判定一定改写。"""
    q = _normalize_question(question)

    if not q:
        return False, "empty question"

    if not _has_usable_history(history):
        return False, "no history"

    if _is_simple_math_question(q):
        return False, "simple math question"

    if _is_system_capability_question(q):
        return False, "system capability question"

    if _has_obvious_follow_up_marker(q):
        return True, "obvious follow-up marker"

    if len(q) <= 20:
        return True, "short ambiguous question with history"

    return False, "not a follow-up question"


def build_rewrite_prompt(question: str, history: list[dict]) -> str:
    """构造历史依赖判断与查询改写提示词，要求模型只返回 JSON。"""
    recent_messages = []
    for message in (history or [])[-6:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        role_name = "用户" if role == "user" else "助手"
        recent_messages.append(f"{role_name}: {content}")

    history_text = "\n".join(recent_messages) or "无"

    return (
        "你是一个检索问答系统的查询判断与改写器。请判断当前问题是否依赖历史对话。\n"
        "要求：\n"
        "1. 只返回 JSON，不要输出 Markdown 或额外解释。\n"
        "2. 不要回答用户问题，只判断和改写问题。\n"
        "3. 如果当前问题依赖历史，将它改写成独立、完整、适合知识库检索的问题，"
        "need_rewrite=true。\n"
        "4. 如果当前问题本身语义完整、不依赖历史，必须保留原问题，"
        "need_rewrite=false。\n"
        "5. 输出必须是合法 JSON。需要改写时示例："
        "{\"need_rewrite\":true,"
        "\"rewritten_question\":\"改写后的完整问题\","
        "\"reason\":\"判断依据\"}；"
        "不需要改写时示例："
        "{\"need_rewrite\":false,"
        "\"rewritten_question\":\"原问题\","
        "\"reason\":\"当前问题语义完整，不依赖历史\"}。\n\n"
        f"历史对话：\n{history_text}\n\n"
        f"当前问题：{question}"
    )


def _parse_rewrite_json(output: str) -> dict:
    """解析改写模型返回的 JSON 对象。"""
    text = (output or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("query rewrite model did not return JSON")
        data = json.loads(text[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("query rewrite output is not a JSON object")

    rewritten_question = str(data.get("rewritten_question") or "").strip()
    if not rewritten_question:
        raise ValueError("query rewrite output missing rewritten_question")

    need_rewrite = data.get("need_rewrite")
    if not isinstance(need_rewrite, bool):
        raise ValueError("query rewrite output has invalid need_rewrite")

    return {
        "need_rewrite": need_rewrite,
        "rewritten_question": rewritten_question,
        "reason": str(data.get("reason") or "").strip(),
    }


def _query_rewrite_llm_configured() -> bool:
    """判断追问改写 LLM 是否具备最小调用配置。"""
    return bool(
        Settings.QUERY_REWRITE_API_KEY
        and Settings.QUERY_REWRITE_API_URL
        and Settings.QUERY_REWRITE_MODEL_NAME
    )


def call_query_rewrite_llm(prompt: str, trace_id: str = "") -> str:
    """调用追问改写模型并返回原始文本。"""
    headers = {"Content-Type": "application/json"}
    if Settings.QUERY_REWRITE_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.QUERY_REWRITE_API_KEY}"

    payload = {
        "model": Settings.QUERY_REWRITE_MODEL_NAME,
        "temperature": Settings.QUERY_REWRITE_TEMPERATURE,
        "max_tokens": Settings.QUERY_REWRITE_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    logger.info(
        f"[trace_id={trace_id}] [query_rewrite_llm] "
        f"url={Settings.QUERY_REWRITE_API_URL} "
        f"model={Settings.QUERY_REWRITE_MODEL_NAME} "
        f"timeout={Settings.QUERY_REWRITE_TIMEOUT}"
    )

    response = requests.post(
        Settings.QUERY_REWRITE_API_URL,
        headers=headers,
        json=payload,
        timeout=Settings.QUERY_REWRITE_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def rewrite_query_with_history(
    question: str,
    history: list[dict],
    trace_id: str = "",
) -> dict:
    """判断问题是否依赖历史并按需改写；明显追问可在 LLM 不可用时降级。"""
    original_question = (question or "").strip()
    base_result = {
        "original_question": original_question,
        "rewritten_question": original_question,
        "history_used": False,
        "rewritten": False,
        "need_check": False,
        "need_rewrite": False,
        "reason": "",
        "rewrite_llm_used": False,
        "error": "",
    }

    if not Settings.ENABLE_QUERY_REWRITE:
        return {
            **base_result,
            "reason": "query rewrite disabled",
        }

    need_check, reason = should_check_rewrite_with_history(
        original_question,
        history,
    )

    if not need_check:
        return {
            **base_result,
            "reason": reason,
        }

    obvious_follow_up = _has_obvious_follow_up_marker(original_question)

    if not _query_rewrite_llm_configured():
        if obvious_follow_up:
            rewritten_question = _fallback_rewrite(original_question, history)
            fallback_reason = "fallback rewrite without llm"
        else:
            rewritten_question = original_question
            fallback_reason = (
                "ambiguous question but query rewrite llm is not configured"
                if reason == "short ambiguous question with history"
                else "query rewrite llm is not configured"
            )
        return {
            **base_result,
            "need_check": True,
            "need_rewrite": bool(rewritten_question != original_question),
            "rewritten_question": rewritten_question,
            "history_used": bool(rewritten_question != original_question),
            "rewritten": bool(rewritten_question != original_question),
            "reason": fallback_reason,
            "rewrite_llm_used": False,
            "error": "query rewrite llm is not configured",
        }

    try:
        prompt = build_rewrite_prompt(original_question, history)
        output = call_query_rewrite_llm(prompt, trace_id=trace_id)
        parsed = _parse_rewrite_json(output)
        need_rewrite = parsed["need_rewrite"]
        rewritten_question = (
            parsed["rewritten_question"]
            if need_rewrite
            else original_question
        )

        return {
            **base_result,
            "need_check": True,
            "need_rewrite": need_rewrite,
            "rewritten_question": rewritten_question,
            "history_used": need_rewrite,
            "rewritten": rewritten_question != original_question,
            "reason": parsed.get("reason") or reason,
            "rewrite_llm_used": True,
            "error": "",
        }

    except Exception as e:
        error = str(e)
        logger.warning(
            f"[trace_id={trace_id}] [query_rewrite] fallback_after_error "
            f"error={error[:300]}"
        )
        rewritten_question = (
            _fallback_rewrite(original_question, history)
            if obvious_follow_up
            else original_question
        )

        return {
            **base_result,
            "need_check": True,
            "need_rewrite": bool(rewritten_question != original_question),
            "rewritten_question": rewritten_question,
            "history_used": bool(rewritten_question != original_question),
            "rewritten": bool(rewritten_question != original_question),
            "reason": (
                "fallback rewrite after llm error"
                if obvious_follow_up
                else "ambiguous question unchanged after query rewrite llm error"
            ),
            "rewrite_llm_used": True,
            "error": error,
        }
