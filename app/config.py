"""应用配置模型和惰性配置入口。

配置解析由 pydantic-settings 负责；模块导入本身不会读取 ``.env``、
实例化配置、创建目录或连接任何外部资源。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
WEB_SEARCH_TOOL_TIMEOUT_SECONDS = 10
WEB_FETCH_TOOL_TIMEOUT_SECONDS = 30
_TOOL_PERMISSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)

LangGraphCheckpointer = Literal["disabled", "memory", "sqlite"]
WebSearchSafeSearch = Literal["on", "moderate", "off"]
ToolPermissions = Annotated[tuple[str, ...], NoDecode]


def project_path(path: Path) -> Path:
    """展开用户目录，并把相对路径解析到项目根目录。"""
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else BASE_DIR / expanded


def secret_value(value: SecretStr | None) -> str | None:
    """仅在模型传输边界解开 SecretStr。"""
    return value.get_secret_value() if value is not None else None


def validate_trusted_web_proxy_url(value: str) -> str:
    """校验显式委托网络策略的 HTTP(S) 受信代理 URL。"""
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError("WEB_TRUSTED_PROXY_URL 包含非法字符")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise ValueError("WEB_TRUSTED_PROXY_URL 格式无效") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "WEB_TRUSTED_PROXY_URL 必须是无 path、query、fragment 的 HTTP(S) URL"
        )
    return value


def resolve_web_proxy(
    *,
    trusted_proxy_mode: bool,
    trusted_proxy_url: str | None,
    fallback_proxy: str | None,
) -> str | None:
    """受信模式开启时统一覆盖工具原有的独立代理配置。"""
    if not trusted_proxy_mode:
        return fallback_proxy
    if not trusted_proxy_url:
        raise ValueError(
            "WEB_TRUSTED_PROXY_MODE=true 时必须配置 WEB_TRUSTED_PROXY_URL"
        )
    return validate_trusted_web_proxy_url(trusted_proxy_url)


class AppSettings(BaseSettings):
    """完整、类型化且不可变的应用配置快照。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
        frozen=True,
        populate_by_name=True,
    )

    # RAG 与索引
    rag_file_path: Path = BASE_DIR / "data"
    index_store_dir: Path = BASE_DIR / "data" / "manifest"
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection_name: str = "mini_rag_chunks"
    top_k: int = Field(default=5, ge=1, le=10)
    decompose_top_k: int = Field(default=10, ge=1, le=20)
    chunk_size: int = 600
    chunk_overlap: int = 120
    max_context_chars: int = 2500
    vector_threshold: float = 0.35
    auto_rebuild_on_startup: bool = False

    # 嵌入与重排
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    model_cache_dir_override: Path | None = Field(
        default=None,
        validation_alias="MODEL_CACHE_DIR",
    )
    hf_hub_cache: Path | None = None
    hf_home: Path | None = None
    sentence_transformers_home: Path | None = None

    # 主回答模型
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    model_name: str = "deepseek-chat"
    llm_api_url: str = "https://api.deepseek.com/v1/chat/completions"
    temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # 查询改写
    enable_query_rewrite: bool = True
    query_rewrite_api_key: SecretStr | None = None
    query_rewrite_model_name: str | None = None
    query_rewrite_api_url: str | None = None
    query_rewrite_timeout: int = 10
    query_rewrite_max_tokens: int = 256
    query_rewrite_temperature: float = 0.0

    # 查询分解
    enable_query_decompose: bool = True
    decompose_api_key: SecretStr | None = None
    decompose_model_name: str = "qwen2.5:7b"
    decompose_api_url: str = "http://localhost:11434/v1/chat/completions"
    decompose_timeout: int = 15
    decompose_max_tokens: int = 512

    # 结果评判
    use_llm_judge: bool = False
    judge_api_key: SecretStr | None = None
    judge_model: str = ""
    judge_api_url: str | None = None
    judge_timeout: int | None = None
    judge_max_retries: int = 1
    judge_max_tokens: int = 2048

    # Agent 与图运行时
    agent_max_iterations: int = Field(default=6, ge=1, le=100)
    agent_max_tool_calls: int = Field(default=10, ge=1, le=1000)
    agent_max_tool_errors: int = Field(default=3, ge=1, le=1000)
    agent_max_consecutive_same_tool_calls: int = Field(
        default=2,
        ge=1,
        le=100,
    )
    agent_max_consecutive_same_tool_errors: int = Field(
        default=2,
        ge=1,
        le=100,
    )
    agent_max_total_cost_ms: int = Field(
        default=120_000,
        ge=1,
        le=3_600_000,
    )
    langgraph_checkpointer: LangGraphCheckpointer = "disabled"
    langgraph_checkpoint_db_path: Path = (
        BASE_DIR / "data" / "langgraph_checkpoints.sqlite3"
    )
    enable_agent_trace_store: bool = False

    # 工具运行时与治理
    tool_runtime_max_workers: int = Field(default=8, ge=1)
    tool_retry_base_delay_ms: int = Field(default=100, ge=0)
    tool_retry_max_delay_ms: int = Field(default=2000, ge=0)
    tool_retry_jitter_ratio: float = Field(
        default=0.2,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    tool_default_principal_id: str = "chat_api"
    tool_default_permissions: ToolPermissions = ("knowledge_base:read",)
    tool_audit_enabled: bool = True
    tool_audit_path: Path = BASE_DIR / "logs" / "tool_audit.jsonl"
    tool_audit_max_depth: int = Field(default=6, ge=1)
    tool_audit_max_items: int = Field(default=20, ge=1)
    tool_audit_max_string_chars: int = Field(default=500, ge=1)
    tool_audit_max_serialized_chars: int = Field(default=5000, ge=1)

    # Web 工具
    web_trusted_proxy_mode: bool = False
    web_trusted_proxy_url: str | None = None
    web_search_region: str = "us-en"
    web_search_safesearch: WebSearchSafeSearch = "moderate"
    web_search_timeout: int = Field(
        default=8,
        ge=1,
        le=WEB_SEARCH_TOOL_TIMEOUT_SECONDS,
    )
    web_search_proxy: str | None = None
    web_fetch_connect_timeout: float = Field(
        default=3,
        gt=0,
        allow_inf_nan=False,
    )
    web_fetch_read_timeout: float = Field(
        default=10,
        gt=0,
        allow_inf_nan=False,
    )
    web_fetch_max_response_bytes: int = Field(default=2_000_000, ge=1)
    web_fetch_max_redirects: int = Field(default=5, ge=0)
    web_fetch_user_agent: str = "rag-agent-qa-system/1.0"
    web_fetch_proxy: str | None = None
    web_pdf_max_response_bytes: int = Field(default=10_000_000, ge=1)
    web_pdf_max_document_pages: int = Field(default=1000, ge=1)
    web_pdf_tool_timeout_seconds: float = Field(
        default=45,
        gt=0,
        allow_inf_nan=False,
    )

    # 存储
    sqlite_db_path: Path = BASE_DIR / "data" / "app.sqlite"

    # 可观测性
    metrics_path: Path = BASE_DIR / "logs" / "metrics.jsonl"
    bad_cases_path: Path = BASE_DIR / "logs" / "bad_cases.jsonl"

    def __init__(self, **values: Any) -> None:
        """拒绝未知显式参数，同时允许 dotenv 中存在其他应用变量。"""
        unknown = {
            name
            for name in values
            if not name.startswith("_") and name not in self.__class__.model_fields
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"未知 AppSettings 初始化字段: {names}")
        super().__init__(**values)

    @field_validator(
        "rag_file_path",
        "index_store_dir",
        "chroma_persist_dir",
        "langgraph_checkpoint_db_path",
        "tool_audit_path",
        "sqlite_db_path",
        "metrics_path",
        "bad_cases_path",
        mode="before",
    )
    @classmethod
    def _normalize_project_path(cls, value: str | Path) -> Path:
        """将必填路径展开并解析到项目根目录。"""
        return project_path(Path(value))

    @field_validator(
        "model_cache_dir_override",
        "hf_hub_cache",
        "hf_home",
        "sentence_transformers_home",
        mode="before",
    )
    @classmethod
    def _normalize_optional_project_path(
        cls,
        value: str | Path | None,
    ) -> Path | None:
        """将空路径视为未配置，否则展开并解析到项目根目录。"""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return project_path(Path(value))

    @field_validator(
        "llm_api_key",
        "query_rewrite_api_key",
        "decompose_api_key",
        "judge_api_key",
        mode="before",
    )
    @classmethod
    def _normalize_optional_secret(
        cls,
        value: str | SecretStr | None,
    ) -> str | SecretStr | None:
        """将环境变量中的空字符串密钥规范为未配置。"""
        if isinstance(value, str) and not value:
            return None
        return value

    @field_validator(
        "top_k",
        "decompose_top_k",
        "chunk_size",
        "chunk_overlap",
        "max_context_chars",
        "vector_threshold",
        "temperature",
        "llm_max_tokens",
        "llm_timeout",
        "llm_max_retries",
        "query_rewrite_timeout",
        "query_rewrite_max_tokens",
        "query_rewrite_temperature",
        "decompose_timeout",
        "decompose_max_tokens",
        "judge_timeout",
        "judge_max_retries",
        "judge_max_tokens",
        "agent_max_iterations",
        "agent_max_tool_calls",
        "agent_max_tool_errors",
        "agent_max_consecutive_same_tool_calls",
        "agent_max_consecutive_same_tool_errors",
        "agent_max_total_cost_ms",
        "tool_runtime_max_workers",
        "tool_retry_base_delay_ms",
        "tool_retry_max_delay_ms",
        "tool_retry_jitter_ratio",
        "tool_audit_max_depth",
        "tool_audit_max_items",
        "tool_audit_max_string_chars",
        "tool_audit_max_serialized_chars",
        "web_search_timeout",
        "web_fetch_connect_timeout",
        "web_fetch_read_timeout",
        "web_fetch_max_response_bytes",
        "web_fetch_max_redirects",
        "web_pdf_max_response_bytes",
        "web_pdf_max_document_pages",
        "web_pdf_tool_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_numbers(cls, value: object) -> object:
        """避免 Python ``bool`` 被静默当成整数 0/1。"""
        if isinstance(value, bool):
            raise ValueError("数值配置不能使用布尔值")
        return value

    @field_validator(
        "web_trusted_proxy_url",
        "web_search_proxy",
        "web_fetch_proxy",
        mode="before",
    )
    @classmethod
    def _normalize_optional_proxy(cls, value: object) -> str | None:
        """去除可选代理两端空白，并将空字符串规范为未配置。"""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Web 代理必须为空或字符串")
        return value.strip() or None

    @field_validator("web_search_region", "tool_default_principal_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """去除必填文本两端空白，并拒绝空值。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("配置值不能为空")
        return normalized

    @field_validator("web_search_safesearch", mode="before")
    @classmethod
    def _normalize_safesearch(cls, value: object) -> object:
        """统一安全搜索配置的大小写与两端空白。"""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("langgraph_checkpointer", mode="before")
    @classmethod
    def _normalize_checkpointer(cls, value: object) -> object:
        """统一检查点后端名称，并将空字符串恢复为禁用状态。"""
        if not isinstance(value, str):
            return value
        return value.strip().lower() or "disabled"

    @field_validator("tool_default_permissions", mode="before")
    @classmethod
    def _parse_permissions(cls, value: object) -> tuple[str, ...]:
        """把逗号分隔文本或序列解析为不可变权限元组。"""
        if isinstance(value, str):
            return tuple(
                item.strip() for item in value.split(",") if item.strip()
            )
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("TOOL_DEFAULT_PERMISSIONS 必须是权限序列")

    @model_validator(mode="after")
    def _validate_configuration(self) -> AppSettings:
        """校验需要多个配置字段共同满足的约束。"""
        if self.tool_retry_max_delay_ms < self.tool_retry_base_delay_ms:
            raise ValueError(
                "TOOL_RETRY_MAX_DELAY_MS 必须大于或等于 "
                "TOOL_RETRY_BASE_DELAY_MS"
            )
        if (
            self.web_fetch_connect_timeout + self.web_fetch_read_timeout
            >= WEB_FETCH_TOOL_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "WEB_FETCH_CONNECT_TIMEOUT 与 WEB_FETCH_READ_TIMEOUT 之和必须小于 "
                f"web_fetch 工具总体超时 {WEB_FETCH_TOOL_TIMEOUT_SECONDS} 秒"
            )
        if (
            self.web_fetch_connect_timeout + self.web_fetch_read_timeout
            >= self.web_pdf_tool_timeout_seconds
        ):
            raise ValueError(
                "WEB_PDF_TOOL_TIMEOUT_SECONDS 必须大于连接与读取超时之和，"
                "以便为 PDF 解析保留预算"
            )
        if (
            not self.web_fetch_user_agent.strip()
            or "\r" in self.web_fetch_user_agent
            or "\n" in self.web_fetch_user_agent
        ):
            raise ValueError("WEB_FETCH_USER_AGENT 必须是无换行的非空字符串")

        resolve_web_proxy(
            trusted_proxy_mode=self.web_trusted_proxy_mode,
            trusted_proxy_url=self.web_trusted_proxy_url,
            fallback_proxy=self.web_search_proxy,
        )
        if not self.tool_default_permissions:
            raise ValueError(
                "TOOL_DEFAULT_PERMISSIONS 至少需要一个精确权限"
            )
        for permission in self.tool_default_permissions:
            if not _TOOL_PERMISSION_PATTERN.fullmatch(permission):
                raise ValueError(
                    "TOOL_DEFAULT_PERMISSIONS 必须使用 resource:action 格式"
                )
        return self

    @property
    def model_cache_dir(self) -> Path:
        """按既有优先级返回 embedding/rerank 模型缓存目录。"""
        if self.model_cache_dir_override is not None:
            return self.model_cache_dir_override
        if self.hf_hub_cache is not None:
            return self.hf_hub_cache
        if self.hf_home is not None:
            return self.hf_home / "hub"
        if self.sentence_transformers_home is not None:
            return self.sentence_transformers_home
        return BASE_DIR / ".cache" / "huggingface" / "hub"

    @property
    def query_rewrite_api_key_resolved(self) -> SecretStr | None:
        """返回查询改写专用密钥，未配置时回退到主模型密钥。"""
        return self.query_rewrite_api_key or self.llm_api_key

    @property
    def query_rewrite_model_name_resolved(self) -> str:
        """返回查询改写模型名，未配置时回退到主模型名。"""
        if self.query_rewrite_model_name is None:
            return self.model_name
        return self.query_rewrite_model_name

    @property
    def query_rewrite_api_url_resolved(self) -> str:
        """返回查询改写接口地址，未配置时回退到主模型地址。"""
        if self.query_rewrite_api_url is None:
            return self.llm_api_url
        return self.query_rewrite_api_url

    @property
    def judge_api_key_resolved(self) -> SecretStr | None:
        """返回评判模型专用密钥，未配置时回退到主模型密钥。"""
        return self.judge_api_key or self.llm_api_key

    @property
    def judge_api_url_resolved(self) -> str:
        """返回评判模型接口地址，未配置时回退到主模型地址。"""
        if self.judge_api_url is None:
            return self.llm_api_url
        return self.judge_api_url

    @property
    def judge_timeout_resolved(self) -> int:
        """返回评判模型超时，未配置时回退到主模型超时。"""
        if self.judge_timeout is None:
            return self.llm_timeout
        return self.judge_timeout

    @property
    def resolved_web_search_proxy(self) -> str | None:
        """按受信代理策略解析 Web 搜索最终使用的代理。"""
        return resolve_web_proxy(
            trusted_proxy_mode=self.web_trusted_proxy_mode,
            trusted_proxy_url=self.web_trusted_proxy_url,
            fallback_proxy=self.web_search_proxy,
        )

    @property
    def resolved_web_fetch_proxy(self) -> str | None:
        """按受信代理策略解析 Web 抓取最终使用的代理。"""
        return resolve_web_proxy(
            trusted_proxy_mode=self.web_trusted_proxy_mode,
            trusted_proxy_url=self.web_trusted_proxy_url,
            fallback_proxy=self.web_fetch_proxy,
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """惰性构造并复用进程级配置快照。"""
    return AppSettings()


def clear_settings_cache() -> None:
    """显式清理配置缓存；高层资源缓存由各自模块负责。"""
    get_settings.cache_clear()


def ensure_model_cache_dir(settings: AppSettings) -> Path:
    """在显式模型初始化阶段创建并返回模型缓存目录。"""
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.model_cache_dir


def validate_rag_environment(settings: AppSettings) -> None:
    """校验 RAG 启动所需的外部路径，但不创建任何资源。"""
    if not settings.rag_file_path.exists():
        raise ValueError(f"RAG_FILE_PATH 不存在: {settings.rag_file_path}")


def validate_startup_environment(settings: AppSettings) -> None:
    """校验 API 启动外部前提，不访问网络或初始化运行时。"""
    if settings.llm_api_key is None:
        raise ValueError("缺少 LLM_API_KEY 或 DEEPSEEK_API_KEY，请检查 .env")
    validate_rag_environment(settings)

    file_paths = [("SQLITE_DB_PATH", settings.sqlite_db_path)]
    if settings.langgraph_checkpointer == "sqlite":
        file_paths.append(
            (
                "LANGGRAPH_CHECKPOINT_DB_PATH",
                settings.langgraph_checkpoint_db_path,
            )
        )
    if settings.tool_audit_enabled:
        file_paths.append(("TOOL_AUDIT_PATH", settings.tool_audit_path))
    for name, path in file_paths:
        if path.exists() and path.is_dir():
            raise ValueError(f"{name} 必须指向文件，不能是目录: {path}")


__all__ = [
    "AppSettings",
    "BASE_DIR",
    "clear_settings_cache",
    "ensure_model_cache_dir",
    "get_settings",
    "project_path",
    "resolve_web_proxy",
    "secret_value",
    "validate_rag_environment",
    "validate_startup_environment",
]
