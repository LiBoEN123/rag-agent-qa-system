"""Query-decomposition rules, LLM transport, and output parsing."""

import json
import re

import requests

from app.config import Settings
from app.prompts import DECOMPOSE_PROMPT


def default_decompose_result(reason: str = "query decomposition skipped") -> dict:
    return {"need_decompose": False, "reason": reason, "sub_queries": []}


def should_decompose_query(question: str) -> tuple[bool, str]:
    query = (question or "").strip()
    if not query:
        return False, "empty query"
    if query.replace("？", "?").count("?") >= 2:
        return True, "multiple question marks"
    keywords = [
        "分别", "同时", "以及", "对比", "区别", "新旧", "变化", "调整",
        "修订", "流程和条件", "条件和材料", "入口和时间",
    ]
    for keyword in keywords:
        if keyword in query:
            return True, f"multi-intent keyword: {keyword}"
    if len(set(re.findall(r"(?:19|20)\d{2}", query))) >= 2:
        return True, "compares multiple years"
    version_pattern = (
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|现行|原规定|"
        r"以前|之前|修订前|修订后|调整前|调整后).{0,20}"
        r"(对比|区别|变化|调整|修订|相比|不同|差异)|"
        r"(对比|区别|变化|调整|修订|相比|不同|差异).{0,20}"
        r"(新版|旧版|新政策|旧政策|新规|旧规|新办法|旧办法|现行|原规定|"
        r"以前|之前|修订前|修订后|调整前|调整后)"
    )
    if re.search(version_pattern, query):
        return True, "compares policy versions"
    intents = (
        "怎么|如何|什么|哪些|是否|能否|有没有|多少|哪里|入口|时间|"
        "条件|材料|流程|要求|办法"
    )
    if re.search(
        rf"({intents}).{{0,30}}(，|,|；|;|、|和|及|并且|同时).{{0,30}}({intents})",
        query,
    ):
        return True, "multiple explicit intents"
    if len(query) <= 30:
        return False, "short single-intent query"
    return False, "single-intent query"


def parse_decompose_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def compact_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_decompose_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError(f"拆分模型返回值必须是 dict，实际为 {type(result)}")
    need_decompose = parse_decompose_bool(result.get("need_decompose", False))
    reason = compact_text(result.get("reason", ""))
    if not need_decompose:
        return default_decompose_result(reason or "decompose model decided single query")
    raw_queries = result.get("sub_queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError("拆分模型返回的 sub_queries 必须是 list")
    sub_queries = []
    for item in raw_queries[:5]:
        if not isinstance(item, dict):
            continue
        sub_question = compact_text(item.get("sub_question", ""))
        search_query = compact_text(item.get("search_query", ""))
        if not sub_question or not search_query:
            continue
        sub_queries.append(
            {
                "id": item.get("id", len(sub_queries) + 1),
                "sub_question": sub_question,
                "search_query": search_query,
                "must_answer": parse_decompose_bool(item.get("must_answer", True)),
            }
        )
    if not sub_queries:
        return default_decompose_result("decompose model returned no valid sub_queries")
    return {
        "need_decompose": True,
        "reason": reason or "decompose model decided multi-query retrieval",
        "sub_queries": sub_queries,
    }


def compact_error_preview(text: str, max_len: int = 500) -> str:
    preview = re.sub(r"\s+", " ", (text or "")).strip()
    return preview[:max_len] + "..." if len(preview) > max_len else preview


def strip_json_code_fence(content: str) -> str:
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", content).strip()


def extract_first_json_object(content: str) -> str:
    start = content.find("{")
    if start < 0:
        raise ValueError("拆分模型未返回 JSON 对象")
    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    raise ValueError("拆分模型返回的 JSON 对象不完整，可能被 max_tokens 截断")


def parse_decompose_llm_json(content: str) -> dict:
    content = strip_json_code_fence(content)
    try:
        result = json.loads(content, strict=False)
    except json.JSONDecodeError:
        try:
            result = json.loads(extract_first_json_object(content), strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM 返回的不是合法 JSON: {compact_error_preview(content)}"
            ) from exc
    return normalize_decompose_result(result)


def decompose_query_with_llm(question: str):
    headers = {"Content-Type": "application/json"}
    if Settings.DECOMPOSE_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.DECOMPOSE_API_KEY}"
    response = requests.post(
        Settings.DECOMPOSE_API_URL,
        headers=headers,
        json={
            "model": Settings.DECOMPOSE_MODEL_NAME,
            "messages": [
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": Settings.DECOMPOSE_MAX_TOKENS,
        },
        timeout=Settings.DECOMPOSE_TIMEOUT,
    )
    response.raise_for_status()
    return parse_decompose_llm_json(
        response.json()["choices"][0]["message"]["content"]
    )
