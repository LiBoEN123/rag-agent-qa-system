"""RAG 答案生成模块。

负责上下文组装、回答模型调用、JSON 解析、引用绑定和置信度估算。
"""

import json
import time
from typing import List

import requests

from app.config import Settings
from app.llm.errors import ModelCallError
from app.llm.factory import get_answer_model
from app.llm.gateway import ModelGateway
from app.prompts import SYSTEM_PROMPT_B, USER_PROMPT, NO_ANSWER_FALLBACK
from app.rag.retrieval import clamp01, make_snippet, safe_float
from app.common.logging import get_logger

logger = get_logger('chat_service')


# =========================
# 上下文构造
# =========================
def build_context(retrieved_chunks, max_chars: int | None=None):
    """把检索结果整理成 LLM 提示词中的 evidence context。"""
    if max_chars is None:
        max_chars = Settings.MAX_CONTEXT_CHARS

    parts = []
    total = 0

    for item in retrieved_chunks:
        snippet = item["text"]
        block = (
            f"[{item['reference_id']}]\n"
            f"file_name:{item['file_name']}\n"
            f"score:{item['hybrid_score']}\n"
            f"chunk_id:{item['chunk_id']}\n"
            f"snippet:{snippet}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)


# =========================
# 模型调用
# =========================
def build_llm_header() -> dict:
    """构造调用回答模型所需的 HTTP 请求头。"""
    headers = {
        "Content-Type": "application/json"
    }

    if Settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.LLM_API_KEY}"

    return headers


def get_http_error_message(status_code: int) -> str:
    """把 HTTP 状态码转换成用户能看懂的提示。"""

    if status_code == 400:
        return "模型请求参数错误，请检查 model、messages、max_tokens 等配置。"

    if status_code == 401:
        return "模型调用未授权，请检查 API Key 是否正确。"

    if status_code == 403:
        return "模型调用被拒绝，请检查账号权限或模型访问权限。"

    if status_code == 404:
        return "模型接口地址或模型名称不存在，请检查 LLM_API_URL 和 MODEL_NAME。"

    if status_code == 429:
        return "模型请求过于频繁或额度不足，请稍后重试或检查账户额度。"

    if status_code >= 500:
        return "模型服务端异常，可能是模型服务暂时不可用。"

    return f"模型服务返回异常状态码：{status_code}"


def _build_answer_messages(question: str, context: str) -> list[dict[str, str]]:
    """使用原有 Prompt 内容构造统一消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_B},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                context=context,
                question=question,
            ),
        },
    ]


def _legacy_call_llm(question: str, context: str, trace_id: str = "") -> str:
    """保留原 requests 调用、重试和 OpenAI-compatible 解析逻辑。"""
    model_name = Settings.MODEL_NAME
    api_url = Settings.LLM_API_URL
    temperature = Settings.TEMPERATURE
    max_tokens = Settings.LLM_MAX_TOKENS

    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": _build_answer_messages(question, context),
    }

    headers = build_llm_header()

    total_attempts = Settings.LLM_MAX_RETRIES + 1

    for attempt in range(total_attempts):
        try:
            logger.info(
                f"[trace_id={trace_id}] [llm] attempt={attempt + 1}/{total_attempts}, "
                f"url={Settings.LLM_API_URL}, "
                f"model={Settings.MODEL_NAME}, "
                f"timeout={Settings.LLM_TIMEOUT}"
            )

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=Settings.LLM_TIMEOUT
            )

            # 优先处理非 200 状态码
            if response.status_code != 200:
                error_message = get_http_error_message(response.status_code)

                # 5xx可以重试
                if response.status_code >= 500:
                    raise  requests.HTTPError(
                        f"{error_message} response={response.text[:300]}"
                    )

                # 400 / 401 / 403 / 404 / 429 这类通常重试没意义，直接抛出
                raise ModelCallError(
                    user_message=error_message,
                    detail=response.text[:500],
                    provider="legacy",
                    model_name=model_name,
                    retryable=False,
                )

            # 解析JSON
            try:
                data = response.json()
            except Exception as e:
                raise ModelCallError(
                    user_message="模型返回内容不是合法 JSON。",
                    detail=str(e),
                    provider="legacy",
                    model_name=model_name,
                    retryable=False,
                )

            # 解析 OpenAI-compatible 格式
            try:
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise ModelCallError(
                    user_message="模型返回格式不符合预期，缺少 choices/message/content 字段。",
                    detail=str(e),
                    provider="legacy",
                    model_name=model_name,
                    retryable=False,
                )

        except ModelCallError:
            # 已经是可读错误，直接抛给上层
            raise

        except requests.exceptions.Timeout as e:
            last_error = e
            error_message = "模型调用超时，请检查模型服务是否响应过慢，或适当调大 LLM_TIMEOUT。"

        except requests.exceptions.ConnectionError as e:
            last_error = e
            error_message = "无法连接模型服务，请检查 LLM_API_URL 是否正确，以及模型服务是否已启动。"

        except requests.exceptions.HTTPError as e:
            last_error = e
            error_message = "模型服务端异常，请稍后重试。"

        except requests.exceptions.RequestException as e:
            last_error = e
            error_message = "模型请求失败，请检查模型服务和网络配置。"

            # 走到这里说明是可重试类错误
        if attempt < Settings.LLM_MAX_RETRIES:
            logger.warning(
                f"[trace_id={trace_id}] [llm] 调用失败，准备重试 attempt={attempt + 1}/{total_attempts}, "
                f"error={error_message}"
            )
            time.sleep(Settings.LLM_RETRY_SLEEP)
            continue

        logger.error(
            f"[trace_id={trace_id}] [llm] 调用最终失败 attempts={total_attempts}, "
            f"error={error_message}"
        )

        raise ModelCallError(
            user_message=error_message,
            detail=str(last_error),
            provider="legacy",
            model_name=model_name,
            retryable=True,
        )


def call_llm(question: str, context: str, trace_id: str = "") -> str:
    """通过统一 Gateway 调用回答模型，必要时回退原 requests 客户端。"""
    messages = _build_answer_messages(question, context)
    mode = (Settings.LLM_CLIENT_MODE or "langchain").strip().lower()
    langchain_model = None
    if mode != "legacy":
        try:
            langchain_model = get_answer_model()
        except ModelCallError as exc:
            logger.warning(
                "[trace_id=%s] [llm] langchain_factory_unavailable "
                "error_type=%s fallback=true",
                trace_id,
                type(exc).__name__,
            )

    def legacy_invoke(
        _messages: list[dict[str, str]],
        *,
        trace_id: str = "",
    ) -> str:
        return _legacy_call_llm(question, context, trace_id=trace_id)

    gateway = ModelGateway(
        role="answer",
        langchain_model=langchain_model,
        legacy_invoke=legacy_invoke,
        client_mode=mode,
    )
    return gateway.invoke_text(messages, trace_id=trace_id)


# =========================
# 模型输出处理
# =========================
def parse_llm_output(llm_output: str):
    """解析模型返回的 JSON，提取 answer 和 references。"""
    llm_output = llm_output.strip()

    if llm_output.startswith("```"):
        llm_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(llm_output)
        answer = data.get("answer", NO_ANSWER_FALLBACK)
        references = data.get("references", [])
        if not isinstance(references, list):
            references = []

        return {
            "ok": True,
            "answer": answer,
            "references": references
        }
    except Exception:
        return {
            "ok": False,
            "answer": NO_ANSWER_FALLBACK,
            "references": []
        }


def format_reference(retrieved_chunks, selected_reference: list):
    """将模型选择的引用编号映射回真实 chunk 元数据。"""
    ref_map = {item["reference_id"]: item for item in retrieved_chunks}

    final_refs = []
    for ref_id in selected_reference:
        if ref_id in ref_map:
            item = ref_map[ref_id]
            final_refs.append({
                "reference_id": item["reference_id"],
                "file_name": item["file_name"],
                "source_path": item["source_path"],
                "chunk_id": item["chunk_id"],
                "position": item.get("position"),
                "snippet": make_snippet(item["text"]),
                "score": round(float(item.get("hybrid_score", 0.0)), 4),
                "rerank_score":round(float(item.get("rerank_score", 0.0)), 4)
            })

    return final_refs

def chunk_evidence_score(item: dict) -> float:
    """把单个 chunk 的多路检索信号压成 0 到 1 的证据强度。"""
    vector_score = clamp01(safe_float(item.get("vector_score")))
    rerank_score = clamp01(safe_float(item.get("rerank_score")))
    hybrid_score = clamp01(safe_float(item.get("hybrid_score", item.get("score", 0.0))))
    lexical_hit = 1.0 if safe_float(item.get("bm25_score")) > 0 else 0.0

    weighted_score = (
        0.50 * vector_score
        + 0.30 * rerank_score
        + 0.15 * hybrid_score
        + 0.05 * lexical_hit
    )

    return round(
        clamp01(
            max(
                weighted_score,
                0.85 * vector_score,
                0.75 * rerank_score if (hybrid_score >= 0.20 or lexical_hit) else 0.0,
            )
        ),
        4
    )


def estimate_retrieval_confidence(retrieved_chunks: List[dict]) -> float:
    """根据最终候选证据质量估算检索置信度。"""
    candidates = [
        item
        for item in retrieved_chunks
        if isinstance(item, dict)
    ]
    if not candidates:
        return 0.0

    evidence_scores = sorted(
        (chunk_evidence_score(item) for item in candidates),
        reverse=True
    )
    top_score = evidence_scores[0]
    second_score = evidence_scores[1] if len(evidence_scores) > 1 else 0.0
    score_gap = clamp01(top_score - second_score) if len(evidence_scores) > 1 else 0.0

    supporting_count = sum(
        1
        for score in evidence_scores[:5]
        if score >= 0.35
    )
    support_ratio = clamp01(supporting_count / 3)

    retrieval_score = 0.70 * top_score + 0.20 * support_ratio + 0.10 * score_gap
    return round(clamp01(retrieval_score), 3)


def estimate_answer_confidence(
    retrieved_chunks: List[dict],
    final_references: List[dict],
    status: str,
    retrieval_confidence: float | None = None
) -> float:
    """结合检索质量和引用支撑度估算最终回答置信度。"""
    if status != "OK" or not retrieved_chunks or not final_references:
        return 0.0

    if retrieval_confidence is None:
        retrieval_confidence = estimate_retrieval_confidence(retrieved_chunks)
    retrieval_confidence = clamp01(retrieval_confidence)

    reference_ids = {
        str(ref.get("reference_id", ""))
        for ref in final_references
        if isinstance(ref, dict)
    }
    cited_chunks = [
        item
        for item in retrieved_chunks
        if str(item.get("reference_id", "")) in reference_ids
    ]

    cited_scores = [
        chunk_evidence_score(item)
        for item in cited_chunks
    ]
    citation_quality = (
        sum(cited_scores) / len(cited_scores)
        if cited_scores
        else 0.0
    )
    reference_ratio = clamp01(
        len(final_references) / max(1, min(len(retrieved_chunks), 5))
    )

    answer_score = (
        0.60 * retrieval_confidence
        + 0.25 * citation_quality
        + 0.15 * reference_ratio
    )
    answer_score = min(answer_score, retrieval_confidence + 0.10)
    return round(clamp01(answer_score), 3)
