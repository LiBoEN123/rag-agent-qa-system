"""Knowledge ingestion 的 canonical URI 与稳定 identity 算法。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from app.common.redaction import DEFAULT_SENSITIVE_KEYS, normalize_sensitive_key


IDENTITY_VERSION = 1
_ID_HEX_LENGTH = 64
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_SENSITIVE_QUERY_KEYS = DEFAULT_SENSITIVE_KEYS | {
    "key",
    "sig",
    "signature",
    "signed_url",
    "x_amz_credential",
    "x_amz_signature",
}
_SOURCE_TYPES = {"local_file", "upload", "github", "web", "firecrawl"}


def _source_type_value(source_type: object) -> str:
    value = getattr(source_type, "value", source_type)
    if not isinstance(value, str) or value not in _SOURCE_TYPES:
        raise ValueError(f"不支持的 Knowledge source_type: {value!r}")
    return value


def _reject_unsafe_uri(raw_uri: str) -> str:
    uri = raw_uri.strip()
    if not uri:
        raise ValueError("source_uri 不能为空")
    if len(uri) > 2048:
        raise ValueError("source_uri 不能超过 2048 个字符")
    if _CONTROL_CHARACTER_PATTERN.search(uri):
        raise ValueError("source_uri 不能包含控制字符")
    return uri


def _canonicalize_file_uri(raw_uri: str) -> str:
    if raw_uri.lower().startswith("file:"):
        parts = urlsplit(raw_uri)
        if parts.query or parts.fragment:
            raise ValueError("file source_uri 不能包含 query 或 fragment")
        if parts.username or parts.password:
            raise ValueError("source_uri 不能包含凭据")
        if parts.netloc and parts.netloc.lower() != "localhost":
            raise ValueError("仅支持本地 file source_uri")
        path = Path(unquote(parts.path))
    else:
        path = Path(raw_uri)

    if not path.is_absolute():
        raise ValueError("local_file source_uri 必须使用绝对路径或 file URI")
    return path.absolute().as_uri()


def _canonicalize_http_uri(raw_uri: str) -> str:
    parts = urlsplit(raw_uri)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("web/firecrawl source_uri 必须使用 http 或 https")
    if not parts.hostname:
        raise ValueError("web/firecrawl source_uri 必须包含 host")
    if parts.username or parts.password:
        raise ValueError("source_uri 不能包含用户名或密码")

    for key, _value in parse_qsl(parts.query, keep_blank_values=True):
        if normalize_sensitive_key(key) in _SENSITIVE_QUERY_KEYS:
            raise ValueError("source_uri query 不能包含凭据或签名参数")

    host = parts.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("source_uri port 无效") from exc
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    return urlunsplit((scheme, host, parts.path or "/", parts.query, ""))


def _canonicalize_github_uri(raw_uri: str) -> str:
    parts = urlsplit(raw_uri)
    if parts.scheme.lower() != "github" or not parts.hostname:
        raise ValueError(
            "github source_uri 必须使用 github://owner/repository/path 格式"
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("github source_uri 不能包含凭据、query 或 fragment")

    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2:
        raise ValueError("github source_uri 必须同时包含 repository 和 file path")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("github source_uri path 不能包含相对路径段")

    owner = parts.hostname.lower()
    repository = segments[0].lower()
    file_path = "/".join(segments[1:])
    return f"github://{owner}/{repository}/{file_path}"


def _canonicalize_upload_uri(raw_uri: str) -> str:
    parts = urlsplit(raw_uri)
    if parts.scheme.lower() != "upload" or not parts.hostname:
        raise ValueError("upload source_uri 必须使用 upload://namespace/id 格式")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("upload source_uri 不能包含凭据、query 或 fragment")
    path = parts.path.rstrip("/")
    if not path or path == "/":
        raise ValueError("upload source_uri 必须包含 upload id")
    return urlunsplit(("upload", parts.hostname.lower(), path, "", ""))


def canonicalize_source_uri(source_type: object, source_uri: str) -> str:
    """按 source type 做保守 canonicalization，不删除业务 query。"""
    if not isinstance(source_uri, str):
        raise TypeError("source_uri 必须是字符串")
    raw_uri = _reject_unsafe_uri(source_uri)
    source_value = _source_type_value(source_type)
    if source_value == "local_file":
        return _canonicalize_file_uri(raw_uri)
    if source_value in {"web", "firecrawl"}:
        return _canonicalize_http_uri(raw_uri)
    if source_value == "github":
        return _canonicalize_github_uri(raw_uri)
    return _canonicalize_upload_uri(raw_uri)


def _sha256(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_source_id(source_type: object, canonical_source_uri: str) -> str:
    """为同一逻辑来源生成与进程随机种子无关的稳定 ID。"""
    source_value = _source_type_value(source_type)
    digest = _sha256(
        f"knowledge-source:v{IDENTITY_VERSION}",
        source_value,
        canonical_source_uri,
    )
    return f"src_{digest}"


def build_document_uid(source_id: str) -> str:
    """在独立 namespace 中为当前一源一文档语义生成稳定文档 ID。"""
    digest = _sha256(f"knowledge-document:v{IDENTITY_VERSION}", source_id)
    return f"doc_{digest}"


def build_content_hash(normalized_content: str) -> str:
    """以 SHA-256 标识实际进入候选合同的完整规范化正文。"""
    return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()


def build_candidate_id(source_id: str, content_hash: str) -> str:
    """用来源与内容共同生成幂等候选 ID。"""
    digest = _sha256(
        f"ingestion-candidate:v{IDENTITY_VERSION}",
        source_id,
        content_hash,
    )
    return f"cand_{digest}"


def is_stable_id(value: str, prefix: str) -> bool:
    """校验公开 identity 的前缀与 SHA-256 hex 长度。"""
    expected_length = len(prefix) + _ID_HEX_LENGTH
    return (
        len(value) == expected_length
        and value.startswith(prefix)
        and all(character in "0123456789abcdef" for character in value[len(prefix):])
    )


__all__ = [
    "IDENTITY_VERSION",
    "build_candidate_id",
    "build_content_hash",
    "build_document_uid",
    "build_source_id",
    "canonicalize_source_uri",
    "is_stable_id",
]
