import json
import time
from typing import List

import requests

from app.config import Settings
from app.prompts import SYSTEM_PROMPT_A, SYSTEM_PROMPT_B, USER_PROMPT, NO_ANSWER_FALLBACK
from app.rag_retrieval import clamp01, make_snippet
from app.untils import get_logger

logger = get_logger('chat_service')


def build_context(retrieved_chunks, max_chars: int | None=None):
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

class LLMCallError(Exception):
    """模型调用失败时使用的统一异常。"""
    def __init__(self, user_message: str, detail: str=""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail

def build_llm_header() -> dict:
    headers = {
        "Content-Type": "application/json"
    }

    if Settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {Settings.LLM_API_KEY}"

    return headers

def  get_http_error_message(status_code: int) -> str:
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

def call_llm(question: str, context: str, trace_id : str = ""):
    model_name = Settings.MODEL_NAME
    api_url = Settings.LLM_API_URL
    temperature = Settings.TEMPERATURE
    max_tokens = Settings.LLM_MAX_TOKENS

    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context=context,
                    question=question
                )
            }
        ]
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
                raise LLMCallError(
                    user_message=error_message,
                    detail=response.text[:500]
                )

            # 解析JSON
            try:
                data = response.json()
            except Exception as e:
                raise LLMCallError(
                    user_message="模型返回内容不是合法 JSON。",
                    detail=str(e)
                )

            # 解析 OpenAI-compatible 格式
            try:
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                raise LLMCallError(
                    user_message="模型返回格式不符合预期，缺少 choices/message/content 字段。",
                    detail=str(e)
                )

        except LLMCallError:
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
            error_message = f"模型请求失败：{e}"

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

        raise LLMCallError(
            user_message=error_message,
            detail=str(last_error)
        )



def parse_llm_output(llm_output: str):
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
                "score": round(float(item.get("hybrid_score", 0.0)), 4)
            })

    return final_refs


def estimate_confidence(
    retrieved_chunks: List[dict],
    final_references: List[dict],
    status: str
) -> float:
    if status != "OK" or not retrieved_chunks:
        return 0.0

    top1_vector = clamp01(retrieved_chunks[0].get("vector_score", 0.0))

    scope_gap = 0.0
    if len(retrieved_chunks) > 1:
        scope_gap = clamp01(
            retrieved_chunks[0].get("vector_score", 0.0) -
            retrieved_chunks[1].get("vector_score", 0.0)
        )

    ref_ratio = clamp01(len(final_references) / max(1, min(len(retrieved_chunks), 5)))

    confidence = 0.65 * top1_vector + 0.20 * scope_gap + 0.15 * ref_ratio
    return round(clamp01(confidence), 3)