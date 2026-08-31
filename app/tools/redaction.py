"""工具兼容入口；权威脱敏实现位于框架无关的 ``app.common``。"""

from app.common.redaction import (
    DEFAULT_SENSITIVE_KEYS,
    REDACTED,
    TRUNCATED,
    normalize_sensitive_key,
    redact_sensitive_data,
    redact_sensitive_text,
)


__all__ = [
    "DEFAULT_SENSITIVE_KEYS",
    "REDACTED",
    "TRUNCATED",
    "normalize_sensitive_key",
    "redact_sensitive_data",
    "redact_sensitive_text",
]
