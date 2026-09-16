"""递归脱敏与体量限制测试。"""

import json

from app.common.redaction import REDACTED, redact_sensitive_data


def test_redaction_handles_nested_sensitive_keys_case_insensitively():
    result = redact_sensitive_data(
        {
            "username": "alice",
            "password": "123456",
            "nested": {
                "API_KEY": "secret-key",
                "access-token": "abc",
            },
        }
    )
    assert result["username"] == "alice"
    assert result["password"] == REDACTED
    assert result["nested"]["API_KEY"] == REDACTED
    assert result["nested"]["access-token"] == REDACTED


def test_redaction_marks_all_truncation_dimensions_and_cycles():
    cyclic = {"text": "abcdefgh", "items": list(range(8))}
    cyclic["self"] = cyclic
    result = redact_sensitive_data(
        cyclic,
        max_depth=3,
        max_items=3,
        max_string_chars=4,
        max_serialized_chars=500,
    )
    serialized = json.dumps(result, ensure_ascii=False)
    assert "truncated" in serialized
    assert "abcd" in serialized
    assert "circular_reference" in serialized


def test_redaction_limits_total_serialized_size():
    result = redact_sensitive_data(
        {"value": "x" * 1000},
        max_serialized_chars=80,
    )
    assert "truncated" in str(result)
    assert len(result) <= 80

