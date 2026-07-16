"""日志配置模块。

统一创建控制台与文件日志处理器，为各业务模块提供一致的日志格式。
"""

import logging
import sys
from urllib.parse import unquote


class DecodedUvicornAccessFilter(logging.Filter):
    """仅在访问日志展示阶段解码 URL，不修改实际请求。"""

    marker = "decoded_uvicorn_access"

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True

        client_addr, method, path, http_version, status_code = args
        decoded_path = unquote(str(path), errors="replace")
        safe_path = "".join(
            char if char.isprintable() else f"\\x{ord(char):02x}"
            for char in decoded_path
        )
        record.args = (
            client_addr,
            method,
            safe_path,
            http_version,
            status_code,
        )
        return True


def configure_uvicorn_access_log() -> None:
    """让 Uvicorn access logger 以可读文本展示百分号编码的 URL。"""
    access_logger = logging.getLogger("uvicorn.access")
    if any(
        getattr(log_filter, "marker", "") == DecodedUvicornAccessFilter.marker
        for log_filter in access_logger.filters
    ):
        return
    access_logger.addFilter(DecodedUvicornAccessFilter())


configure_uvicorn_access_log()

def get_logger(name:str = "app"):
    """创建或复用统一格式的 stdout 日志记录器。"""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel("INFO")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel("INFO")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger




















