import json
import re

import requests

from app.config import Settings
from app.untils import get_logger


logger = get_logger("chat_service")

FOLLOW_UP_MARKERS = [
    "那",
    "那这个",
    "那它",
    "这个",
    "那个",
    "它",
    "上面",
    "前面",
    "刚才",
    "继续",
    "还有呢",
    "休学创业呢",
    "这种情况呢",
]

REFERENCE_MARKERS = ["这个", "那个", "它", "上面", "刚才", "前面"]
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
QUESTION_CUES = [
    "多少",
    "几",
    "什么",
    "如何",
    "怎么",
    "为什么",
    "是否",
    "是不是",
    "吗",
    "哪",
]


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


def _looks_standalone_question(question: str) -> bool:
    """粗略判断短问题是否已经具备独立语义。"""
    q = _normalize_question(question)
    if not q:
        return False

    if q.startswith(("那", "这个", "那个", "它", "上面", "前面", "刚才", "继续")):
        return False

    if any(marker in q for marker in REFERENCE_MARKERS):
        return False

    return any(cue in q for cue in QUESTION_CUES)


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


def should_rewrite_with_history(question: str, history: list[dict]) -> tuple[bool, str]:
    """根据轻量规则判断当前问题是否需要结合历史改写。"""
    q = _normalize_question(question)

    if not q:
        return False, "empty question"

    if not _has_usable_history(history):
        return False, "no history"

    if _is_simple_math_question(q):
        return False, "simple math question"

    if _is_system_capability_question(q):
        return False, "system capability question"

    if q.startswith("那"):
        return True, "question starts with follow-up marker"

    if any(marker in q for marker in REFERENCE_MARKERS):
        return True, "question contains reference marker"

    if q.endswith(("呢?", "呢")) and len(q) <= 18:
        return True, "short question ends with follow-up particle"

    if len(q) <= 15 and not _looks_standalone_question(q):
        return True, "short follow-up question with history"

    return False, "not a follow-up question"


def build_rewrite_prompt(question: str, history: list[dict]) -> str:
    """构造追问改写提示词，要求模型只返回 JSON。"""
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
        "你是一个检索问答系统的查询改写器。请根据历史对话，把当前用户追问改写成"
        "独立、完整、可直接用于知识库检索的问题。\n"
        "要求：\n"
        "1. 只返回 JSON，不要输出 Markdown 或额外解释。\n"
        "2. 不要回答问题，只改写问题。\n"
        "3. 如果历史不足以改写，尽量保留原问题并说明原因。\n"
        "4. JSON 格式必须为："
        "{\"rewritten_question\":\"完整问题\",\"reason\":\"为什么这样改写\"}\n\n"
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

    return {
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


def rewrite_query_with_history(question: str, history: list[dict], trace_id: str = "") -> dict:
    """结合历史对追问做轻量改写，失败时回退到安全拼接策略。"""
    original_question = (question or "").strip()
    base_result = {
        "original_question": original_question,
        "rewritten_question": original_question,
        "history_used": False,
        "rewritten": False,
        "reason": "",
        "rewrite_llm_used": False,
        "error": "",
    }

    if not Settings.ENABLE_QUERY_REWRITE:
        return {
            **base_result,
            "reason": "query rewrite disabled",
        }

    need_rewrite, reason = should_rewrite_with_history(original_question, history)
    if not need_rewrite:
        return {
            **base_result,
            "reason": reason,
        }

    if not _query_rewrite_llm_configured():
        rewritten_question = _fallback_rewrite(original_question, history)
        return {
            **base_result,
            "rewritten_question": rewritten_question,
            "history_used": bool(rewritten_question != original_question),
            "rewritten": bool(rewritten_question != original_question),
            "reason": "fallback rewrite without llm",
            "rewrite_llm_used": False,
            "error": "query rewrite llm is not configured",
        }

    try:
        prompt = build_rewrite_prompt(original_question, history)
        output = call_query_rewrite_llm(prompt, trace_id=trace_id)
        parsed = _parse_rewrite_json(output)
        rewritten_question = parsed["rewritten_question"]

        return {
            **base_result,
            "rewritten_question": rewritten_question,
            "history_used": True,
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
        rewritten_question = _fallback_rewrite(original_question, history)

        return {
            **base_result,
            "rewritten_question": rewritten_question,
            "history_used": bool(rewritten_question != original_question),
            "rewritten": bool(rewritten_question != original_question),
            "reason": "fallback rewrite after llm error",
            "rewrite_llm_used": True,
            "error": error,
        }
