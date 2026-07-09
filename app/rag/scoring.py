"""Small score-normalization helpers shared across RAG stages."""

import math


def safe_float(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if math.isfinite(score) else default
