

"""工具运行时错误定义。

提供稳定的错误码，以及允许工具显式声明错误类别和可重试性的异常。
"""

from enum import Enum


class ToolErrorCode(str, Enum):
    """工具运行时可对外返回的稳定错误码。"""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"

    TIMEOUT = "timeout"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    RATE_LIMITED = "rate_limited"

    EXECUTION_ERROR = "execution_error"
    INVALID_RESULT = "invalid_result"
    OUTPUT_VALIDATION_ERROR = "output_validation_error"

    RUNTIME_UNAVAILABLE = "runtime_unavailable"


class ToolExecutionError(Exception):
    """允许工具实现显式声明错误类别及其是否可重试。"""

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode = ToolErrorCode.EXECUTION_ERROR,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = ToolErrorCode(code)
        self.retryable = retryable


__all__ = ["ToolErrorCode", "ToolExecutionError"]
