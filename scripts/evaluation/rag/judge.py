"""RAG 语义评测的可选 LLM judge 适配与结果规范化。

用途：构造严格提示词、经统一模型网关调用 judge，并把自由输出收敛为稳定结构。
输入：问题、标准答案、关键点/禁用点、模型答案、证据和 judge 配置。
输出：规范化的关键点覆盖或禁用信息判断，以及可审计的原始响应摘要。
边界：judge 关闭或不可用时采用保守结果；最终分数仍由 ``rag.evaluation`` 计算。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.config import get_settings, secret_value
from app.common.logging import get_logger
from app.llm.factory import get_judge_model
from app.llm.gateway import ModelGateway
from app.prompts import FORBIDDEN_JUDGE_SYSTEM_PROMPT, KEY_POINT_JUDGE_SYSTEM_PROMPT
from scripts.evaluation.rag.common import json_dumps, point_texts, warn


logger = get_logger("rag_evaluator")


ALLOWED_KEY_POINT_STATUSES = {"covered", "partial", "missing", "contradicted"}


@dataclass(frozen=True)
class LLMJudgeConfig:
    """LLM judge 调用配置。enabled=False 时使用 no-judge 保守兜底。"""

    enabled: bool
    model: str
    api_url: str
    api_key: str
    timeout: int
    max_retries: int
    max_tokens: int
    temperature: float = 0.0


class LLMJudgeError(Exception):
    """LLM judge 调用或解析失败，但不应中断整体评估。"""


def make_judge_config(
    use_llm_judge: bool | None = None,
    judge_model: str | None = None,
    judge_api_url: str | None = None,
    judge_api_key: str | None = None,
) -> LLMJudgeConfig:
    """从 CLI 和 Settings 合成 judge 配置；没有 JUDGE_MODEL 时不启用 judge。"""
    settings = get_settings()
    enabled = (
        settings.use_llm_judge
        if use_llm_judge is None
        else use_llm_judge
    )
    model = (
        judge_model if judge_model is not None else settings.judge_model
    ).strip()
    api_url = (
        judge_api_url
        if judge_api_url is not None
        else settings.judge_api_url_resolved
    ).strip()
    api_key = (
        judge_api_key
        if judge_api_key is not None
        else secret_value(settings.judge_api_key_resolved) or ""
    )

    if enabled and not model:
        warn("USE_LLM_JUDGE is enabled but JUDGE_MODEL is empty; LLM semantic scores will use conservative no-judge fallback")
        enabled = False
    if enabled and not api_url:
        warn("USE_LLM_JUDGE is enabled but JUDGE_API_URL is empty; LLM semantic scores will use conservative no-judge fallback")
        enabled = False

    return LLMJudgeConfig(
        enabled=enabled,
        model=model,
        api_url=api_url,
        api_key=api_key or "",
        timeout=settings.judge_timeout_resolved,
        max_retries=settings.judge_max_retries,
        max_tokens=settings.judge_max_tokens,
    )


def compact_error_preview(text: str, max_len: int = 500) -> str:
    """将模型原始输出压成单行预览，避免日志被换行打断。"""
    preview = re.sub(r"\s+", " ", (text or "")).strip()
    return preview[:max_len] + "..." if len(preview) > max_len else preview


def strip_json_code_fence(content: str) -> str:
    """去掉模型可能返回的 Markdown JSON 代码块。"""
    content = (content or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def extract_first_json_object(content: str) -> str:
    """从模型输出中截取第一个完整 JSON 对象字符串。"""
    start = content.find("{")
    if start < 0:
        raise ValueError("LLM judge did not return a JSON object")

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
                return content[start:index + 1]

    raise ValueError("LLM judge JSON object is incomplete")


def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
    """解析 LLM judge 输出，兼容代码块和 JSON 前后夹杂文本。"""
    content = strip_json_code_fence(response_text)
    try:
        parsed = json.loads(content, strict=False)
    except json.JSONDecodeError:
        json_object = extract_first_json_object(content)
        try:
            parsed = json.loads(json_object, strict=False)
        except json.JSONDecodeError as exc:
            preview = compact_error_preview(content)
            raise ValueError(f"LLM judge returned invalid JSON: {preview}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM judge JSON root must be an object")
    return parsed


def normalize_status(status: Any) -> str:
    """将 judge 状态规范化为四种允许值。"""
    raw = str(status or "").strip().lower()
    mapping = {
        "覆盖": "covered",
        "已覆盖": "covered",
        "部分": "partial",
        "部分覆盖": "partial",
        "缺失": "missing",
        "未覆盖": "missing",
        "矛盾": "contradicted",
        "冲突": "contradicted",
    }
    raw = mapping.get(raw, raw)
    return raw if raw in ALLOWED_KEY_POINT_STATUSES else "missing"


def parse_bool(value: Any) -> bool:
    """宽松解析 judge 返回的布尔字段。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "hit", "命中", "是"}


def build_key_point_judge_prompt(
    question: str,
    gold_answer: str,
    gold_key_points: List[str],
    model_answer: str,
    evidence: List[str] | None = None,
    category: str = "",
    expected_behavior: str = "",
) -> str:
    """构造逐关键点语义判断 prompt。"""
    payload = {
        "question": question,
        "category": category,
        "expected_behavior": expected_behavior,
        "gold_answer": gold_answer,
        "gold_key_points": gold_key_points,
        "model_answer": model_answer,
        "retrieval_evidence_optional": evidence or [],
    }
    return (
        "请根据系统规则完成逐条关键点覆盖判断。"
        "注意：检索证据只用于理解背景，不能替模型回答补全。\n\n"
        + json_dumps(payload)
    )


def build_forbidden_judge_prompt(
    question: str,
    model_answer: str,
    forbidden_points: List[str],
    gold_answer: str = "",
    evidence: List[str] | None = None,
) -> str:
    """构造 forbidden 语义判断 prompt。"""
    payload = {
        "question": question,
        "model_answer": model_answer,
        "forbidden_points": forbidden_points,
        "gold_answer_optional": gold_answer,
        "retrieval_evidence_optional": evidence or [],
    }
    return "请根据系统规则判断 forbidden_points 是否被当成事实回答。\n\n" + json_dumps(payload)


def call_judge_llm(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
) -> str:
    """通过统一 Gateway 调用 judge 模型并返回原始 content。"""
    try:
        model = get_judge_model(
            api_key=judge_config.api_key,
            api_url=judge_config.api_url,
            model_name=judge_config.model,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            timeout=judge_config.timeout,
            max_retries=judge_config.max_retries,
        )
        gateway = ModelGateway(role="judge", langchain_model=model)
        logger.info(
            "[judge] trace_id=%s model=%s timeout=%s",
            trace_id,
            judge_config.model,
            judge_config.timeout,
        )
        return gateway.invoke_text(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            trace_id=trace_id,
        )
    except LLMJudgeError:
        raise
    except Exception as exc:
        raise LLMJudgeError(str(exc)) from exc


def call_judge_json(
    system_prompt: str,
    user_prompt: str,
    judge_config: LLMJudgeConfig,
    trace_id: str = "",
    call_judge_llm_fn: Any = None,
) -> Tuple[Dict[str, Any], str]:
    """调用 judge 并解析 JSON；JSON 解析失败时额外重试一次。"""
    if call_judge_llm_fn is None:
        call_judge_llm_fn = call_judge_llm

    parse_errors = []
    raw_text = ""
    for parse_attempt in range(2):
        prompt = user_prompt
        if parse_attempt == 1:
            prompt += (
                "\n\n上一次输出不是合法 JSON。请重新输出严格 JSON："
                "不要 Markdown，不要解释性文字，不要省略双引号。"
            )
        raw_text = call_judge_llm_fn(system_prompt, prompt, judge_config, trace_id=trace_id)
        try:
            return parse_llm_json_response(raw_text), raw_text
        except ValueError as exc:
            parse_errors.append(str(exc))
            logger.warning(
                f"[judge] trace_id={trace_id} JSON parse failed "
                f"attempt={parse_attempt + 1}/2 error={exc}"
            )

    raise LLMJudgeError("; ".join(parse_errors) or "LLM judge JSON parse failed")


def compute_key_point_score(key_point_judgments: List[Dict[str, Any]]) -> float:
    """covered 计 1 分、partial 计 0.5 分，由程序计算关键点召回。"""
    total = len(key_point_judgments)
    if total == 0:
        return 1.0
    covered_count = sum(1 for item in key_point_judgments if item.get("status") == "covered")
    partial_count = sum(1 for item in key_point_judgments if item.get("status") == "partial")
    return (covered_count * 1.0 + partial_count * 0.5) / total


def normalize_key_point_judge_result(raw: Dict[str, Any], gold_key_points: List[str]) -> Dict[str, Any]:
    """按标准关键点顺序规范化 LLM judge 结果，并由程序重新计算计数。"""
    raw_judgments = raw.get("key_point_judgments", [])
    if not isinstance(raw_judgments, list):
        raw_judgments = []

    by_key = {}
    for item in raw_judgments:
        if isinstance(item, dict):
            key = str(item.get("key_point", "")).strip()
            if key:
                by_key[key] = item

    judgments = []
    for idx, key_point in enumerate(gold_key_points):
        raw_item = by_key.get(key_point)
        if raw_item is None and idx < len(raw_judgments) and isinstance(raw_judgments[idx], dict):
            raw_item = raw_judgments[idx]
        if raw_item is None:
            raw_item = {}

        raw_status = raw_item.get("status")
        status = normalize_status(raw_status)
        reason = str(raw_item.get("reason", "") or "").strip()
        if str(raw_status or "").strip().lower() not in ALLOWED_KEY_POINT_STATUSES and status == "missing":
            reason = (reason + "；" if reason else "") + "judge 未返回合法状态，按 missing 处理"

        judgments.append({
            "key_point": key_point,
            "status": status,
            "evidence_from_answer": str(raw_item.get("evidence_from_answer", "") or "").strip(),
            "reason": reason,
        })

    missing_key_points = [j["key_point"] for j in judgments if j["status"] == "missing"]
    partial_key_points = [j["key_point"] for j in judgments if j["status"] == "partial"]
    contradicted_key_points = [j["key_point"] for j in judgments if j["status"] == "contradicted"]
    covered_count = sum(1 for j in judgments if j["status"] == "covered")
    partial_count = len(partial_key_points)
    missing_count = len(missing_key_points)
    contradicted_count = len(contradicted_key_points)

    return {
        "key_point_judgments": judgments,
        "missing_key_points": missing_key_points,
        "partial_key_points": partial_key_points,
        "contradicted_key_points": contradicted_key_points,
        "covered_count": covered_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "contradicted_count": contradicted_count,
        "key_point_recall": compute_key_point_score(judgments),
    }


def normalize_forbidden_judge_result(raw: Dict[str, Any], forbidden_points: List[str]) -> Dict[str, Any]:
    """按 forbidden_points 顺序规范化 LLM forbidden 判断结果。"""
    raw_judgments = raw.get("forbidden_judgments", [])
    if not isinstance(raw_judgments, list):
        raw_judgments = []

    by_key = {}
    for item in raw_judgments:
        if isinstance(item, dict):
            key = str(item.get("forbidden_point", "")).strip()
            if key:
                by_key[key] = item

    judgments = []
    for idx, forbidden_point in enumerate(forbidden_points):
        raw_item = by_key.get(forbidden_point)
        if raw_item is None and idx < len(raw_judgments) and isinstance(raw_judgments[idx], dict):
            raw_item = raw_judgments[idx]
        if raw_item is None:
            raw_item = {}

        judgments.append({
            "forbidden_point": forbidden_point,
            "hit": parse_bool(raw_item.get("hit", False)),
            "answer_span": str(raw_item.get("answer_span", "") or "").strip(),
            "reason": str(raw_item.get("reason", "") or "").strip(),
        })

    return {
        "forbidden_hit": any(item["hit"] for item in judgments),
        "forbidden_judgments": judgments,
    }


def evaluate_key_points_with_llm(
    question: str,
    gold_answer: str,
    gold_key_points: List[Any],
    model_answer: str,
    evidence: List[str] | None = None,
    judge_config: LLMJudgeConfig | None = None,
    category: str = "",
    expected_behavior: str = "",
    qid: str = "",
    call_judge_json_fn: Any = None,
) -> Dict[str, Any]:
    """使用 LLM judge 逐条判断关键点，再由程序计算 key_point_recall。"""
    points = point_texts(gold_key_points)
    if not points:
        return normalize_key_point_judge_result({"key_point_judgments": []}, [])
    if not model_answer.strip():
        judgments = [
            {
                "key_point": point,
                "status": "missing",
                "evidence_from_answer": "",
                "reason": "模型回答为空",
            }
            for point in points
        ]
        return normalize_key_point_judge_result({"key_point_judgments": judgments}, points)
    if judge_config is None or not judge_config.enabled:
        raise LLMJudgeError("LLM judge is not enabled")

    prompt = build_key_point_judge_prompt(
        question=question,
        gold_answer=gold_answer,
        gold_key_points=points,
        model_answer=model_answer,
        evidence=evidence,
        category=category,
        expected_behavior=expected_behavior,
    )
    judge_call = call_judge_json_fn or call_judge_json
    raw, raw_text = judge_call(KEY_POINT_JUDGE_SYSTEM_PROMPT, prompt, judge_config, trace_id=f"{qid}:key_points")
    result = normalize_key_point_judge_result(raw, points)
    result["judge_raw_response"] = raw_text
    return result


def evaluate_forbidden_with_llm(
    question: str,
    model_answer: str,
    forbidden_points: List[Any],
    evidence: List[str] | None = None,
    judge_config: LLMJudgeConfig | None = None,
    gold_answer: str = "",
    qid: str = "",
    call_judge_json_fn: Any = None,
) -> Dict[str, Any]:
    """使用 LLM judge 语义判断 forbidden hit，区分事实性编造和否定性提及。"""
    points = point_texts(forbidden_points)
    if not points:
        return normalize_forbidden_judge_result({"forbidden_judgments": []}, [])
    if not model_answer.strip():
        judgments = [
            {
                "forbidden_point": point,
                "hit": False,
                "answer_span": "",
                "reason": "模型回答为空",
            }
            for point in points
        ]
        return normalize_forbidden_judge_result({"forbidden_judgments": judgments}, points)
    if judge_config is None or not judge_config.enabled:
        raise LLMJudgeError("LLM judge is not enabled")

    prompt = build_forbidden_judge_prompt(
        question=question,
        model_answer=model_answer,
        forbidden_points=points,
        gold_answer=gold_answer,
        evidence=evidence,
    )
    judge_call = call_judge_json_fn or call_judge_json
    raw, raw_text = judge_call(FORBIDDEN_JUDGE_SYSTEM_PROMPT, prompt, judge_config, trace_id=f"{qid}:forbidden")
    result = normalize_forbidden_judge_result(raw, points)
    result["judge_raw_response"] = raw_text
    return result


def unavailable_key_point_result(gold_key_points: List[Any], reason: str) -> Dict[str, Any]:
    """LLM judge 不可用时的保守兜底：不做关键词匹配，关键点按 missing 记录。"""
    points = point_texts(gold_key_points)
    judgments = [
        {
            "key_point": point,
            "status": "missing",
            "evidence_from_answer": "",
            "reason": reason,
        }
        for point in points
    ]
    result = normalize_key_point_judge_result({"key_point_judgments": judgments}, points)
    result["judge_raw_response"] = ""
    return result


def unavailable_forbidden_result(forbidden_points: List[Any], reason: str) -> Dict[str, Any]:
    """LLM judge 不可用时的保守兜底：不做关键词 forbidden 判断。"""
    points = point_texts(forbidden_points)
    judgments = [
        {
            "forbidden_point": point,
            "hit": False,
            "answer_span": "",
            "reason": reason,
        }
        for point in points
    ]
    result = normalize_forbidden_judge_result({"forbidden_judgments": judgments}, points)
    result["judge_raw_response"] = ""
    return result
