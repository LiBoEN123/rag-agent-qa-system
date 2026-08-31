"""框架无关的递归脱敏与安全体量限制。"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from enum import Enum
from typing import Any


REDACTED = "***REDACTED***"
TRUNCATED = "...<truncated>"
DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "authorization",
        "cookie",
        "session_token",
        "private_key",
        "credential",
        "client_secret",
    }
)
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?i)\b(authorization|cookie|password|passwd|secret|token|"
    r"access[\s_-]?token|refresh[\s_-]?token|api[\s_-]?key|"
    r"client[\s_-]?secret)"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_TEXT_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def normalize_sensitive_key(value: object) -> str:
    """把大小写、连字符和空格差异规整为统一敏感键。"""
    return re.sub(r"[-\s]+", "_", str(value).strip().lower())


def redact_sensitive_text(value: str) -> str:
    """从不可信自由文本中移除常见凭据形态。"""
    redacted = _SENSITIVE_TEXT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
        value,
    )
    redacted = _BEARER_TEXT_PATTERN.sub(REDACTED, redacted)
    return _OPENAI_STYLE_KEY_PATTERN.sub(REDACTED, redacted)


def _safe_key(value: object, max_string_chars: int) -> str:
    """返回可序列化且受长度约束的字典键。"""
    key = value if isinstance(value, str) else f"<{type(value).__name__}>"
    if len(key) <= max_string_chars:
        return key
    return key[:max_string_chars] + TRUNCATED


def redact_sensitive_data(
    value: Any,
    *,
    sensitive_keys: Collection[str] = DEFAULT_SENSITIVE_KEYS,
    max_depth: int = 6,
    max_items: int = 20,
    max_string_chars: int = 500,
    max_serialized_chars: int = 5000,
) -> Any:
    """递归脱敏并限制深度、项目数、字符串和最终序列化体量。"""
    limits = (max_depth, max_items, max_string_chars, max_serialized_chars)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in limits):
        raise ValueError("所有脱敏体量限制都必须是大于或等于 1 的整数")

    normalized_sensitive_keys = {
        normalize_sensitive_key(key)
        for key in DEFAULT_SENSITIVE_KEYS.union(set(sensitive_keys))
    }
    active_ids: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            text = redact_sensitive_text(item)
            if len(text) <= max_string_chars:
                return text
            return text[:max_string_chars] + TRUNCATED
        if isinstance(item, bytes):
            return f"<bytes length={len(item)}>"
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if depth >= max_depth:
            return f"<max_depth>{TRUNCATED}"

        identity = id(item)
        if isinstance(item, (Mapping, list, tuple, set, frozenset)):
            if identity in active_ids:
                return f"<circular_reference>{TRUNCATED}"
            active_ids.add(identity)
            try:
                if isinstance(item, Mapping):
                    output: dict[str, Any] = {}
                    entries = list(item.items())
                    for key, nested in entries[:max_items]:
                        safe_key = _safe_key(key, max_string_chars)
                        if normalize_sensitive_key(key) in normalized_sensitive_keys:
                            output[safe_key] = REDACTED
                        else:
                            output[safe_key] = visit(nested, depth + 1)
                    if len(entries) > max_items:
                        output[TRUNCATED] = (
                            f"{len(entries) - max_items} fields omitted"
                        )
                    return output

                sequence = list(item)
                output_items = [
                    visit(nested, depth + 1)
                    for nested in sequence[:max_items]
                ]
                if len(sequence) > max_items:
                    output_items.append(
                        f"{len(sequence) - max_items} items omitted{TRUNCATED}"
                    )
                return output_items
            finally:
                active_ids.remove(identity)

        return f"<{type(item).__name__}>"

    sanitized = visit(value, 0)
    serialized = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) <= max_serialized_chars:
        return sanitized

    if max_serialized_chars <= len(TRUNCATED):
        return TRUNCATED[:max_serialized_chars]
    available = max_serialized_chars - len(TRUNCATED)
    return serialized[:available] + TRUNCATED


__all__ = [
    "DEFAULT_SENSITIVE_KEYS",
    "REDACTED",
    "TRUNCATED",
    "normalize_sensitive_key",
    "redact_sensitive_data",
    "redact_sensitive_text",
]
