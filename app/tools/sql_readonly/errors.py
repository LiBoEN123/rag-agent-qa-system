"""只读 SQL 工具的有限、安全错误分类。"""

from app.tools.errors import ToolErrorCode


class SqlReadonlyError(Exception):
    """携带稳定错误码且保证消息不包含连接凭据的 SQL 错误。"""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


__all__ = ["SqlReadonlyError"]
