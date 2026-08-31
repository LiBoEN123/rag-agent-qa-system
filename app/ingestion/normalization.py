"""Knowledge content 与 metadata 的确定性安全规范化。"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any

from app.common.redaction import redact_sensitive_data, redact_sensitive_text


CONTENT_NORMALIZATION = "nfc-newlines-trailing-whitespace-v1"
METADATA_MAX_DEPTH = 6
METADATA_MAX_KEYS = 100
METADATA_MAX_COLLECTION_ITEMS = 100
METADATA_MAX_KEY_CHARS = 128
METADATA_MAX_STRING_CHARS = 2000
METADATA_MAX_SERIALIZED_BYTES = 32_768


def prepare_content(content: str) -> tuple[str, bool]:
    """先脱敏，再做保守 normalization，并报告是否发生脱敏。"""
    if not isinstance(content, str):
        raise TypeError("content 必须是字符串")
    redacted = redact_sensitive_text(content)
    normalized = unicodedata.normalize("NFC", redacted)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))
    return normalized.rstrip(" \t\n"), redacted != content


def _validate_json_value(
    value: Any,
    *,
    depth: int,
    key_count: list[int],
) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("metadata 浮点数必须是有限值")
        return
    if type(value) is str:
        if len(value) > METADATA_MAX_STRING_CHARS:
            raise ValueError(
                f"metadata 字符串不能超过 {METADATA_MAX_STRING_CHARS} 个字符"
            )
        return
    if depth >= METADATA_MAX_DEPTH:
        raise ValueError(f"metadata 嵌套深度不能超过 {METADATA_MAX_DEPTH}")
    if type(value) is list:
        if len(value) > METADATA_MAX_COLLECTION_ITEMS:
            raise ValueError(
                "metadata 单个列表不能超过 "
                f"{METADATA_MAX_COLLECTION_ITEMS} 项"
            )
        for item in value:
            _validate_json_value(item, depth=depth + 1, key_count=key_count)
        return
    if type(value) is dict:
        if len(value) > METADATA_MAX_COLLECTION_ITEMS:
            raise ValueError(
                "metadata 单个对象不能超过 "
                f"{METADATA_MAX_COLLECTION_ITEMS} 个键"
            )
        key_count[0] += len(value)
        if key_count[0] > METADATA_MAX_KEYS:
            raise ValueError(f"metadata 总键数不能超过 {METADATA_MAX_KEYS}")
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("metadata 的所有 key 必须是字符串")
            if not key or len(key) > METADATA_MAX_KEY_CHARS:
                raise ValueError(
                    "metadata key 必须为非空且不能超过 "
                    f"{METADATA_MAX_KEY_CHARS} 个字符"
                )
            if redact_sensitive_text(key) != key:
                raise ValueError("metadata key 不能包含凭据")
            _validate_json_value(item, depth=depth + 1, key_count=key_count)
        return
    raise TypeError(
        "metadata 仅允许 str/int/float/bool/None/list/dict，"
        f"不允许 {type(value).__name__}"
    )


def prepare_metadata(metadata: Any) -> tuple[dict[str, Any], bool]:
    """拒绝非 JSON 值、施加体量限制，并复用统一规则执行脱敏。"""
    if type(metadata) is not dict:
        raise TypeError("metadata 必须是 dict")
    _validate_json_value(metadata, depth=0, key_count=[0])

    raw_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(raw_json.encode("utf-8")) > METADATA_MAX_SERIALIZED_BYTES:
        raise ValueError(
            "metadata JSON 不能超过 "
            f"{METADATA_MAX_SERIALIZED_BYTES} UTF-8 bytes"
        )

    sanitized = redact_sensitive_data(
        metadata,
        max_depth=METADATA_MAX_DEPTH + 2,
        max_items=METADATA_MAX_COLLECTION_ITEMS + 1,
        max_string_chars=METADATA_MAX_STRING_CHARS,
        max_serialized_chars=METADATA_MAX_SERIALIZED_BYTES * 4,
    )
    if not isinstance(sanitized, dict):
        raise ValueError("metadata 脱敏后必须仍为 JSON object")
    return sanitized, sanitized != metadata


__all__ = [
    "CONTENT_NORMALIZATION",
    "METADATA_MAX_COLLECTION_ITEMS",
    "METADATA_MAX_DEPTH",
    "METADATA_MAX_KEYS",
    "METADATA_MAX_KEY_CHARS",
    "METADATA_MAX_SERIALIZED_BYTES",
    "METADATA_MAX_STRING_CHARS",
    "prepare_content",
    "prepare_metadata",
]
