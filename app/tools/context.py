"""工具调用期间使用的内部上下文。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolInvocationContext:
    """承载不应暴露到工具参数 Schema 的请求级元数据。"""

    trace_id: str = ""
