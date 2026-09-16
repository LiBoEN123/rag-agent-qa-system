"""MCP Server 静态配置 Schema、URL 约束和安全 Header 解析。"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


_SERVER_ALIAS_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_FORBIDDEN_HEADER_NAMES = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class McpConfigurationError(ValueError):
    """表示不含 Secret 值的 MCP 部署配置错误。"""


def validate_mcp_server_url(value: str) -> str:
    """只允许 HTTPS，或明确的本机 Streamable HTTP URL。"""
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError("MCP Server URL 包含非法字符")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ValueError("MCP Server URL 格式无效") from exc
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("MCP Server URL 必须使用 HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MCP Server URL 不允许内嵌凭据")
    if parsed.fragment:
        raise ValueError("MCP Server URL 不允许 fragment")
    if parsed.query:
        raise ValueError("MCP Server URL 不允许 query；认证信息应使用安全 Header 配置")
    if scheme == "http" and hostname not in _LOOPBACK_HOSTS:
        raise ValueError("远程 MCP Server 必须使用 HTTPS")
    return value


def _normalize_header_name(value: str, *, field_name: str) -> str:
    """校验业务 Header 名，并拒绝由 HTTP 传输层控制的高风险字段。"""
    header = value.strip()
    if not _HEADER_NAME_PATTERN.fullmatch(header):
        raise ValueError(f"{field_name} 包含非法 HTTP Header 名")
    if header.casefold() in _FORBIDDEN_HEADER_NAMES:
        raise ValueError(f"{field_name} 包含禁止配置的 HTTP Header")
    return header


def _validate_header_value(value: str, *, field_name: str) -> str:
    """拒绝空值与任何控制字符，防止 Header 注入。"""
    if not value.strip():
        raise ValueError(f"{field_name} 不允许空 Header 值")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} 的 Header 值包含控制字符")
    return value


class McpServerConfig(BaseModel):
    """一个只允许 Streamable HTTP 和精确工具 allowlist 的 Server 配置。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    enabled: bool = True
    transport: Literal["http"]
    url: str
    allowed_tools: tuple[str, ...]
    approval_required_tools: tuple[str, ...] = ()
    knowledge_promotion: dict[str, Literal["github", "firecrawl"]] = Field(
        default_factory=dict
    )
    enabled_for_default_principal: bool = False
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        """去除 URL 首尾空白并应用项目的 MCP 网络安全约束。"""
        return validate_mcp_server_url(value.strip())

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _validate_allowed_tools(cls, value: object) -> tuple[str, ...]:
        """把工具白名单规范化为非空、无重复的合法名称元组。"""
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError("allowed_tools 必须是非空字符串列表")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("allowed_tools 每一项都必须是字符串")
            name = item.strip()
            if not _TOOL_NAME_PATTERN.fullmatch(name):
                raise ValueError("allowed_tools 只允许精确合法工具名")
            if name in seen:
                raise ValueError("allowed_tools 不允许重复工具名")
            seen.add(name)
            normalized.append(name)
        if not normalized:
            raise ValueError("allowed_tools 不能为空")
        return tuple(normalized)

    @field_validator("approval_required_tools", mode="before")
    @classmethod
    def _validate_approval_required_tools(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        """规范化必须经人工审批的精确远端工具名。"""
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError("approval_required_tools 必须是字符串列表")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("approval_required_tools 每一项都必须是字符串")
            name = item.strip()
            if not _TOOL_NAME_PATTERN.fullmatch(name):
                raise ValueError("approval_required_tools 只允许精确合法工具名")
            if name in seen:
                raise ValueError("approval_required_tools 不允许重复工具名")
            seen.add(name)
            normalized.append(name)
        return tuple(normalized)

    @field_validator("headers", mode="before")
    @classmethod
    def _validate_headers(cls, value: object) -> dict[str, str]:
        """校验 gitignored local JSON 中的直接 Secret Header。"""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("headers 必须是 Header 到 Secret 值的映射")
        normalized: dict[str, str] = {}
        seen: set[str] = set()
        for raw_header, raw_value in value.items():
            if not isinstance(raw_header, str) or not isinstance(raw_value, str):
                raise ValueError("headers 的键和值都必须是字符串")
            header = _normalize_header_name(raw_header, field_name="headers")
            key = header.casefold()
            if key in seen:
                raise ValueError("headers 不允许大小写重复的 Header")
            seen.add(key)
            normalized[header] = _validate_header_value(
                raw_value,
                field_name="headers",
            )
        return normalized

    @field_validator("headers_env", mode="before")
    @classmethod
    def _validate_headers_env(cls, value: object) -> dict[str, str]:
        """校验 Header 名与环境变量引用，不接收任何实际凭据值。"""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("headers_env 应为 Header 到环境变量名的映射")
        normalized: dict[str, str] = {}
        seen: set[str] = set()
        for raw_header, raw_env_name in value.items():
            if not isinstance(raw_header, str) or not isinstance(raw_env_name, str):
                raise ValueError("headers_env 的键和值仅接受字符串")
            header = _normalize_header_name(raw_header, field_name="headers_env")
            env_name = raw_env_name.strip()
            key = header.casefold()
            if not _ENV_NAME_PATTERN.fullmatch(env_name):
                raise ValueError("headers_env 的值应为大写环境变量名")
            if key in seen:
                raise ValueError("headers_env 不允许大小写重复的 Header")
            seen.add(key)
            normalized[header] = env_name
        return normalized

    @model_validator(mode="after")
    def _validate_server_policy(self) -> McpServerConfig:
        """校验审批工具子集与 Header 注入边界。"""
        unknown_approval_tools = set(self.approval_required_tools).difference(
            self.allowed_tools
        )
        if unknown_approval_tools:
            raise ValueError("approval_required_tools 必须是 allowed_tools 的子集")
        for name, source in self.knowledge_promotion.items():
            expected = {
                "github": "get_file_contents",
                "firecrawl": "firecrawl_scrape",
            }[source]
            if (
                name != expected
                or name not in self.allowed_tools
                or name in self.approval_required_tools
            ):
                raise ValueError(
                    "knowledge_promotion 仅允许白名单中的 GitHub file / Firecrawl scrape 只读工具"
                )
        direct_names = {header.casefold() for header in self.headers}
        env_names = {header.casefold() for header in self.headers_env}
        if direct_names & env_names:
            raise ValueError("headers 与 headers_env 不允许配置同一个 Header")
        return self


class McpConfig(BaseModel):
    """完整 MCP Client 配置文件合同。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    version: Literal[1]
    servers: dict[str, McpServerConfig]

    @model_validator(mode="after")
    def _validate_servers(self) -> McpConfig:
        """确保至少配置一个 Server，且每个别名都符合名称合同。"""
        if not self.servers:
            raise ValueError("MCP servers 不能为空")
        for alias in self.servers:
            if not _SERVER_ALIAS_PATTERN.fullmatch(alias):
                raise ValueError("MCP Server alias 必须匹配 ^[a-z][a-z0-9_]*$")
            if any(
                source != alias
                for source in self.servers[alias].knowledge_promotion.values()
            ):
                raise ValueError(
                    "knowledge_promotion 必须使用对应的 github / firecrawl Server alias"
                )
        return self


def load_mcp_config(path: Path) -> McpConfig:
    """从部署者指定路径读取配置；错误信息不回显文件内容。"""
    try:
        resolved_path = path.expanduser().resolve()
    except OSError:
        raise McpConfigurationError("MCP Server 配置路径无法解析") from None
    if not resolved_path.is_file():
        raise McpConfigurationError("MCP Server 配置文件不存在或不是普通文件")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError:
        raise McpConfigurationError("MCP Server 配置文件不可读取") from None
    except (UnicodeError, json.JSONDecodeError):
        raise McpConfigurationError("MCP Server 配置文件不是合法 UTF-8 JSON") from None
    try:
        return McpConfig.model_validate(payload)
    except Exception:
        raise McpConfigurationError("MCP Server 配置不符合安全 Schema") from None


def build_mcp_connections(
    config: McpConfig,
    *,
    tool_timeout_seconds: float,
    environ: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """从部署进程环境解析可选 Header 引用，构造 adapter 连接配置。"""
    source = os.environ if environ is None else environ

    connections: dict[str, dict[str, Any]] = {}
    for alias, server in config.servers.items():
        if not server.enabled:
            continue
        headers: dict[str, str] = {
            header: _validate_header_value(
                secret.get_secret_value(),
                field_name="headers",
            )
            for header, secret in server.headers.items()
        }
        for header, env_name in server.headers_env.items():
            value = source.get(env_name)
            if not isinstance(value, str) or not value.strip():
                raise McpConfigurationError(f"MCP Header 环境变量未配置: {env_name}")
            try:
                headers[header] = _validate_header_value(
                    value,
                    field_name="headers_env",
                )
            except ValueError:
                raise McpConfigurationError(
                    f"MCP Header 环境变量值不安全: {env_name}"
                ) from None
        connection: dict[str, Any] = {
            "transport": "http",
            "url": server.url,
            "timeout": tool_timeout_seconds,
        }
        if headers:
            connection["headers"] = headers
        connections[alias] = connection
    return connections


__all__ = [
    "McpConfig",
    "McpConfigurationError",
    "McpServerConfig",
    "build_mcp_connections",
    "load_mcp_config",
    "validate_mcp_server_url",
]
