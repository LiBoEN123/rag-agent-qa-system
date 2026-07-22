"""应用配置模块。

集中读取环境变量、解析项目路径，并提供模型、索引和服务运行参数。
"""

import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
LANGGRAPH_CHECKPOINTER_VALUES = {"disabled", "memory", "sqlite"}
WEB_SEARCH_SAFESEARCH_VALUES = {"on", "moderate", "off"}
WEB_SEARCH_TOOL_TIMEOUT_SECONDS = 10
_TOOL_PERMISSION_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$"
)


def normalize_langgraph_checkpointer(value: str | None) -> str:
    """规整 LangGraph Checkpointer 模式，空值默认关闭。"""
    normalized = (value or "").strip().lower()
    return normalized or "disabled"


def validate_langgraph_checkpointer(value: str | None) -> str:
    """校验并返回规整后的 LangGraph Checkpointer 模式。"""
    normalized = normalize_langgraph_checkpointer(value)
    if normalized not in LANGGRAPH_CHECKPOINTER_VALUES:
        allowed = ", ".join(sorted(LANGGRAPH_CHECKPOINTER_VALUES))
        raise ValueError(
            f"LANGGRAPH_CHECKPOINTER 配置无效: {value!r}，可选值为: {allowed}"
        )
    return normalized


def validate_langgraph_checkpoint_db_path(value: str | Path) -> Path:
    """解析并校验 Checkpoint SQLite 文件路径，但不创建文件或目录。"""
    path = project_path(Path(value).expanduser())
    if path.exists() and path.is_dir():
        raise ValueError(
            "LANGGRAPH_CHECKPOINT_DB_PATH 必须指向 SQLite 文件，"
            f"不能是目录: {path}"
        )
    return path


def env_bool(name: str, default: str = "false") -> bool:
    """从环境变量读取布尔配置，支持常见真值字符串。"""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: str) -> int:
    """读取整数环境变量，并在格式非法时给出明确配置名。"""
    raw_value = os.getenv(name, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 配置无效: {raw_value!r}，必须是整数") from exc


def env_float(name: str, default: str) -> float:
    """读取浮点环境变量，并在格式非法时给出明确配置名。"""
    raw_value = os.getenv(name, default)
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 配置无效: {raw_value!r}，必须是数字") from exc


def normalize_web_search_proxy(value: str | None) -> str | None:
    """把空白 Web Search 代理配置规整为 None。"""
    return (value or "").strip() or None


def validate_agent_loop_config(
    *,
    max_iterations: int,
    max_tool_calls: int,
    max_tool_errors: int,
    max_consecutive_same_tool_calls: int,
    max_consecutive_same_tool_errors: int,
    max_total_cost_ms: int,
) -> None:
    """校验迭代式 Agent 的轮次、错误、循环和累计耗时预算。"""
    values = {
        "AGENT_MAX_ITERATIONS": max_iterations,
        "AGENT_MAX_TOOL_CALLS": max_tool_calls,
        "AGENT_MAX_TOOL_ERRORS": max_tool_errors,
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS": (
            max_consecutive_same_tool_calls
        ),
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS": (
            max_consecutive_same_tool_errors
        ),
        "AGENT_MAX_TOTAL_COST_MS": max_total_cost_ms,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数")
        if value < 1:
            raise ValueError(f"{name} 必须大于或等于 1")
    if max_iterations > 100:
        raise ValueError("AGENT_MAX_ITERATIONS 必须小于或等于 100")
    if max_tool_calls > 1000:
        raise ValueError("AGENT_MAX_TOOL_CALLS 必须小于或等于 1000")
    if max_tool_errors > 1000:
        raise ValueError("AGENT_MAX_TOOL_ERRORS 必须小于或等于 1000")
    if max_consecutive_same_tool_calls > 100:
        raise ValueError(
            "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS 必须小于或等于 100"
        )
    if max_consecutive_same_tool_errors > 100:
        raise ValueError(
            "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS 必须小于或等于 100"
        )
    if max_total_cost_ms > 3_600_000:
        raise ValueError("AGENT_MAX_TOTAL_COST_MS 必须小于或等于 3600000")


def validate_tool_runtime_config(
    *,
    max_workers: int,
    retry_base_delay_ms: int,
    retry_max_delay_ms: int,
    retry_jitter_ratio: float,
) -> None:
    """校验工具运行时线程池与退避参数。"""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ValueError("TOOL_RUNTIME_MAX_WORKERS 必须是整数")
    if max_workers < 1:
        raise ValueError("TOOL_RUNTIME_MAX_WORKERS 必须大于或等于 1")

    if isinstance(retry_base_delay_ms, bool) or not isinstance(
        retry_base_delay_ms,
        int,
    ):
        raise ValueError("TOOL_RETRY_BASE_DELAY_MS 必须是整数")
    if retry_base_delay_ms < 0:
        raise ValueError("TOOL_RETRY_BASE_DELAY_MS 必须大于或等于 0")

    if isinstance(retry_max_delay_ms, bool) or not isinstance(
        retry_max_delay_ms,
        int,
    ):
        raise ValueError("TOOL_RETRY_MAX_DELAY_MS 必须是整数")
    if retry_max_delay_ms < retry_base_delay_ms:
        raise ValueError(
            "TOOL_RETRY_MAX_DELAY_MS 必须大于或等于 "
            "TOOL_RETRY_BASE_DELAY_MS"
        )

    if isinstance(retry_jitter_ratio, bool) or not isinstance(
        retry_jitter_ratio,
        (int, float),
    ):
        raise ValueError("TOOL_RETRY_JITTER_RATIO 必须是数字")
    if not math.isfinite(float(retry_jitter_ratio)) or not (
        0 <= retry_jitter_ratio <= 1
    ):
        raise ValueError("TOOL_RETRY_JITTER_RATIO 必须在 0 到 1 之间")


def validate_tool_runtime_settings(settings: Any) -> None:
    """从 Settings 类读取并校验工具运行时配置。"""
    validate_tool_runtime_config(
        max_workers=settings.TOOL_RUNTIME_MAX_WORKERS,
        retry_base_delay_ms=settings.TOOL_RETRY_BASE_DELAY_MS,
        retry_max_delay_ms=settings.TOOL_RETRY_MAX_DELAY_MS,
        retry_jitter_ratio=settings.TOOL_RETRY_JITTER_RATIO,
    )


def validate_web_search_config(
    *,
    region: object,
    safesearch: object,
    timeout: object,
    proxy: object,
) -> None:
    """校验 DDGS 客户端配置，不访问网络或创建客户端。"""
    if not isinstance(region, str) or not region.strip():
        raise ValueError("WEB_SEARCH_REGION 不能为空")
    if not isinstance(safesearch, str) or safesearch not in WEB_SEARCH_SAFESEARCH_VALUES:
        allowed = ", ".join(sorted(WEB_SEARCH_SAFESEARCH_VALUES))
        raise ValueError(
            f"WEB_SEARCH_SAFESEARCH 配置无效: {safesearch!r}，可选值为: {allowed}"
        )
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("WEB_SEARCH_TIMEOUT 必须是正整数")
    if timeout > WEB_SEARCH_TOOL_TIMEOUT_SECONDS:
        raise ValueError(
            "WEB_SEARCH_TIMEOUT 必须小于或等于 web_search 工具总体超时 "
            f"{WEB_SEARCH_TOOL_TIMEOUT_SECONDS} 秒"
        )
    if proxy is not None and not isinstance(proxy, str):
        raise ValueError("WEB_SEARCH_PROXY 必须为空或字符串")


def env_csv(name: str, default: str) -> tuple[str, ...]:
    """读取逗号分隔且保持顺序的非空配置项。"""
    raw_value = os.getenv(name, default)
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def validate_tool_governance_config(
    *,
    default_principal_id: str,
    default_permissions: object,
    audit_path: str | Path,
    max_depth: int,
    max_items: int,
    max_string_chars: int,
    max_serialized_chars: int,
) -> None:
    """校验工具默认主体和审计体量限制。"""
    if not isinstance(default_principal_id, str) or not default_principal_id.strip():
        raise ValueError("TOOL_DEFAULT_PRINCIPAL_ID 不能为空")

    if isinstance(default_permissions, str):
        permissions = tuple(
            item.strip()
            for item in default_permissions.split(",")
            if item.strip()
        )
    elif isinstance(default_permissions, (tuple, list)):
        permissions = tuple(default_permissions)
    else:
        raise ValueError("TOOL_DEFAULT_PERMISSIONS 必须是权限序列")
    if not permissions:
        raise ValueError("TOOL_DEFAULT_PERMISSIONS 至少需要一个精确权限")
    for permission in permissions:
        if not isinstance(permission, str) or not _TOOL_PERMISSION_PATTERN.fullmatch(
            permission.strip()
        ):
            raise ValueError(
                "TOOL_DEFAULT_PERMISSIONS 必须使用 resource:action 格式"
            )

    resolved_audit_path = project_path(Path(audit_path).expanduser())
    if resolved_audit_path.exists() and resolved_audit_path.is_dir():
        raise ValueError("TOOL_AUDIT_PATH 必须指向文件，不能是目录")

    limits = {
        "TOOL_AUDIT_MAX_DEPTH": max_depth,
        "TOOL_AUDIT_MAX_ITEMS": max_items,
        "TOOL_AUDIT_MAX_STRING_CHARS": max_string_chars,
        "TOOL_AUDIT_MAX_SERIALIZED_CHARS": max_serialized_chars,
    }
    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是大于或等于 1 的整数")


def validate_tool_governance_settings(settings: Any) -> None:
    """从 Settings 类读取并校验工具治理配置。"""
    validate_tool_governance_config(
        default_principal_id=settings.TOOL_DEFAULT_PRINCIPAL_ID,
        default_permissions=settings.TOOL_DEFAULT_PERMISSIONS,
        audit_path=settings.TOOL_AUDIT_PATH,
        max_depth=settings.TOOL_AUDIT_MAX_DEPTH,
        max_items=settings.TOOL_AUDIT_MAX_ITEMS,
        max_string_chars=settings.TOOL_AUDIT_MAX_STRING_CHARS,
        max_serialized_chars=settings.TOOL_AUDIT_MAX_SERIALIZED_CHARS,
    )


def project_path(path: Path) -> Path:
    """把相对路径解析为项目根目录下的绝对路径。"""
    return path if path.is_absolute() else BASE_DIR / path


def default_model_cache_dir() -> Path:
    """返回模型权重的本地缓存目录，优先尊重显式环境变量。"""
    if os.getenv("MODEL_CACHE_DIR"):
        return project_path(Path(os.getenv("MODEL_CACHE_DIR")).expanduser())
    if os.getenv("HF_HUB_CACHE"):
        return project_path(Path(os.getenv("HF_HUB_CACHE")).expanduser())
    if os.getenv("HF_HOME"):
        return project_path(Path(os.getenv("HF_HOME")).expanduser()) / "hub"
    if os.getenv("SENTENCE_TRANSFORMERS_HOME"):
        return project_path(Path(os.getenv("SENTENCE_TRANSFORMERS_HOME")).expanduser())
    return BASE_DIR / ".cache" / "huggingface" / "hub"


class Settings:
    RAG_FILE_PATH=Path(os.getenv("RAG_FILE_PATH", str(BASE_DIR / 'data')))
    INDEX_STORE_DIR = Path(os.getenv("INDEX_STORE_DIR", str(BASE_DIR / "data" / "manifest"))) # index manifest directory

    CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma")))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "mini_rag_chunks")

    TOP_K = int(os.getenv("TOP_K", "5"))
    DECOMPOSE_TOP_K = int(os.getenv("DECOMPOSE_TOP_K", "10"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 发散程度参数，越低模型回答越稳，越保守；越高模型越发散，越创造。

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME",
                                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME",
                                  "BAAI/bge-reranker-v2-m3")
    MODEL_CACHE_DIR = default_model_cache_dir()

    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 2500))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

    LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
    LLM_API_URL = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")

    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 2))

    TOOL_RUNTIME_MAX_WORKERS = env_int("TOOL_RUNTIME_MAX_WORKERS", "8")
    TOOL_RETRY_BASE_DELAY_MS = env_int("TOOL_RETRY_BASE_DELAY_MS", "100")
    TOOL_RETRY_MAX_DELAY_MS = env_int("TOOL_RETRY_MAX_DELAY_MS", "2000")
    TOOL_RETRY_JITTER_RATIO = env_float("TOOL_RETRY_JITTER_RATIO", "0.2")

    WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "us-en").strip()
    WEB_SEARCH_SAFESEARCH = os.getenv(
        "WEB_SEARCH_SAFESEARCH", "moderate"
    ).strip().lower()
    WEB_SEARCH_TIMEOUT = env_int("WEB_SEARCH_TIMEOUT", "8")
    WEB_SEARCH_PROXY = normalize_web_search_proxy(os.getenv("WEB_SEARCH_PROXY"))

    TOOL_DEFAULT_PRINCIPAL_ID = os.getenv(
        "TOOL_DEFAULT_PRINCIPAL_ID",
        "chat_api",
    ).strip()
    TOOL_DEFAULT_PERMISSIONS = env_csv(
        "TOOL_DEFAULT_PERMISSIONS",
        "knowledge_base:read",
    )

    TOOL_AUDIT_ENABLED = env_bool("TOOL_AUDIT_ENABLED", "true")
    TOOL_AUDIT_PATH = project_path(
        Path(os.getenv("TOOL_AUDIT_PATH", "logs/tool_audit.jsonl")).expanduser()
    )
    TOOL_AUDIT_MAX_DEPTH = env_int("TOOL_AUDIT_MAX_DEPTH", "6")
    TOOL_AUDIT_MAX_ITEMS = env_int("TOOL_AUDIT_MAX_ITEMS", "20")
    TOOL_AUDIT_MAX_STRING_CHARS = env_int(
        "TOOL_AUDIT_MAX_STRING_CHARS",
        "500",
    )
    TOOL_AUDIT_MAX_SERIALIZED_CHARS = env_int(
        "TOOL_AUDIT_MAX_SERIALIZED_CHARS",
        "5000",
    )

    USE_LLM_JUDGE = env_bool("USE_LLM_JUDGE", "false")
    JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or LLM_API_KEY
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
    JUDGE_API_URL = os.getenv("JUDGE_API_URL", LLM_API_URL)
    JUDGE_TIMEOUT = int(os.getenv("JUDGE_TIMEOUT", os.getenv("LLM_TIMEOUT", "60")))
    JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "1"))
    JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "2048"))

    ENABLE_LLM_ROUTER = env_bool("ENABLE_LLM_ROUTER", "false")
    ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or LLM_API_KEY
    ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME", MODEL_NAME)
    ROUTER_API_URL = os.getenv("ROUTER_API_URL", LLM_API_URL)
    ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "10"))
    ROUTER_MAX_RETRIES = int(os.getenv("ROUTER_MAX_RETRIES", "1"))
    ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "512"))
    ROUTER_TEMPERATURE = float(os.getenv("ROUTER_TEMPERATURE", "0.0"))

    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
    AGENT_MAX_ITERATIONS = env_int("AGENT_MAX_ITERATIONS", "6")
    AGENT_MAX_TOOL_CALLS = env_int("AGENT_MAX_TOOL_CALLS", "10")
    AGENT_MAX_TOOL_ERRORS = env_int("AGENT_MAX_TOOL_ERRORS", "3")
    AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS = env_int(
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS",
        "2",
    )
    AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS = env_int(
        "AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS",
        "2",
    )
    AGENT_MAX_TOTAL_COST_MS = env_int("AGENT_MAX_TOTAL_COST_MS", "120000")
    LANGGRAPH_CHECKPOINTER = normalize_langgraph_checkpointer(
        os.getenv("LANGGRAPH_CHECKPOINTER", "disabled")
    )
    LANGGRAPH_CHECKPOINT_DB_PATH = project_path(
        Path(
            os.getenv(
                "LANGGRAPH_CHECKPOINT_DB_PATH",
                str(BASE_DIR / "data" / "langgraph_checkpoints.sqlite3"),
            )
        ).expanduser()
    )

    ENABLE_AGENT_TRACE_STORE = env_bool("ENABLE_AGENT_TRACE_STORE", "false")
    SQLITE_DB_PATH = project_path(
        Path(
            os.getenv(
                "SQLITE_DB_PATH",
                str(BASE_DIR / "data" / "app.sqlite"),
            )
        ).expanduser()
    )

    ENABLE_QUERY_REWRITE = env_bool("ENABLE_QUERY_REWRITE", "true")
    QUERY_REWRITE_API_KEY = os.getenv("QUERY_REWRITE_API_KEY") or LLM_API_KEY
    QUERY_REWRITE_MODEL_NAME = os.getenv("QUERY_REWRITE_MODEL_NAME", MODEL_NAME)
    QUERY_REWRITE_API_URL = os.getenv("QUERY_REWRITE_API_URL", LLM_API_URL)
    QUERY_REWRITE_TIMEOUT = int(os.getenv("QUERY_REWRITE_TIMEOUT", "10"))
    QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "256"))
    QUERY_REWRITE_TEMPERATURE = float(os.getenv("QUERY_REWRITE_TEMPERATURE", "0.0"))

    DECOMPOSE_API_KEY = os.getenv("DECOMPOSE_API_KEY")
    DECOMPOSE_MODEL_NAME = os.getenv("DECOMPOSE_MODEL_NAME", "qwen2.5:7b")
    DECOMPOSE_API_URL = os.getenv("DECOMPOSE_API_URL", "http://localhost:11434/v1/chat/completions")
    # Enables automatic query decomposition. The decomposition model still decides
    # whether a specific question needs sub-queries.
    ENABLE_QUERY_DECOMPOSE = env_bool("ENABLE_QUERY_DECOMPOSE", "true")
    DECOMPOSE_TIMEOUT = int(os.getenv("DECOMPOSE_TIMEOUT", "15"))
    DECOMPOSE_MAX_TOKENS = int(os.getenv("DECOMPOSE_MAX_TOKENS", "512"))

    VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.35"))

    AUTO_REBUILD_ON_STARTUP = env_bool("AUTO_REBUILD_ON_STARTUP", "false") #是否强制启动就重建

    @classmethod
    def validate(cls):
        """校验关键配置是否存在并可访问。"""
        validate_agent_loop_config(
            max_iterations=cls.AGENT_MAX_ITERATIONS,
            max_tool_calls=cls.AGENT_MAX_TOOL_CALLS,
            max_tool_errors=cls.AGENT_MAX_TOOL_ERRORS,
            max_consecutive_same_tool_calls=(
                cls.AGENT_MAX_CONSECUTIVE_SAME_TOOL_CALLS
            ),
            max_consecutive_same_tool_errors=(
                cls.AGENT_MAX_CONSECUTIVE_SAME_TOOL_ERRORS
            ),
            max_total_cost_ms=cls.AGENT_MAX_TOTAL_COST_MS,
        )
        validate_tool_runtime_settings(cls)
        validate_web_search_config(
            region=cls.WEB_SEARCH_REGION,
            safesearch=cls.WEB_SEARCH_SAFESEARCH,
            timeout=cls.WEB_SEARCH_TIMEOUT,
            proxy=cls.WEB_SEARCH_PROXY,
        )
        validate_tool_governance_settings(cls)
        cls.LANGGRAPH_CHECKPOINTER = validate_langgraph_checkpointer(
            cls.LANGGRAPH_CHECKPOINTER
        )
        if cls.LANGGRAPH_CHECKPOINTER == "sqlite":
            cls.LANGGRAPH_CHECKPOINT_DB_PATH = (
                validate_langgraph_checkpoint_db_path(
                    cls.LANGGRAPH_CHECKPOINT_DB_PATH
                )
            )
        if not cls.LLM_API_KEY:
            raise ValueError("缺少 LLM_API_KEY 或 DEEPSEEK_API_KEY，请检查 .env")
        if not cls.RAG_FILE_PATH.exists():
            raise ValueError(f"RAG_FILE_PATH 不存在: {cls.RAG_FILE_PATH}")

    @classmethod
    def ensure_model_cache_dir(cls) -> Path:
        """创建并返回 embedding/rerank 模型的本地缓存目录。"""
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODEL_CACHE_DIR
