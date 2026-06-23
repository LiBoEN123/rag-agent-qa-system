import logging
import sys

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























