"""安全数值转换。

提供跨检索、生成和门控阶段复用的安全数值转换能力。
"""

import math


def safe_float(value, default: float = 0.0) -> float:
    """安全转换有限浮点数，非法值返回指定默认值。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if math.isfinite(score) else default
