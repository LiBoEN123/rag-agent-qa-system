"""LLM 调用的统一异常定义。"""

from __future__ import annotations

import os
import re


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _redact_secrets(value: object) -> str:
    """从异常详情中移除常见凭据，避免被日志或响应意外暴露。"""
    text = str(value or "")
    text = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", text)
    text = _SECRET_PATTERNS[1].sub(r"\1[REDACTED]", text)
    text = _SECRET_PATTERNS[2].sub("[REDACTED]", text)

    for name, secret in os.environ.items():
        if "API_KEY" not in name.upper() or not secret:
            continue
        text = text.replace(secret, "[REDACTED]")
    return text


class ModelCallError(Exception):
    """保存面向用户的提示和经过脱敏的模型调用调试信息。"""

    def __init__(
        self,
        user_message: str,
        detail: str = "",
        *,
        provider: str = "",
        model_name: str = "",
        retryable: bool = False,
    ) -> None:
        safe_user_message = _redact_secrets(user_message)
        super().__init__(safe_user_message)
        self.user_message = safe_user_message
        self.detail = _redact_secrets(detail)
        self.provider = str(provider)
        self.model_name = str(model_name)
        self.retryable = bool(retryable)
