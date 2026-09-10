"""Uvicorn HTTP access log 的显式适配配置。"""

import logging
from urllib.parse import unquote


class DecodedUvicornAccessFilter(logging.Filter):
    """仅在访问日志展示阶段解码 URL，不修改实际请求。"""

    marker = "decoded_uvicorn_access"

    # 解码访问日志中的请求路径并转义不可打印字符，保留日志记录。
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
    """幂等安装 Uvicorn access log 的 URL 安全解码过滤器。"""
    access_logger = logging.getLogger("uvicorn.access")
    if any(
        getattr(log_filter, "marker", "") == DecodedUvicornAccessFilter.marker
        for log_filter in access_logger.filters
    ):
        return
    access_logger.addFilter(DecodedUvicornAccessFilter())
