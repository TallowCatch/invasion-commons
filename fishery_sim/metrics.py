import numpy as np


def gini(x: np.ndarray) -> float:
    """Gini coefficient for nonnegative values."""
    x = np.asarray(x, dtype=float)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n)


def collapse_probability(collapsed_flags: list[bool]) -> float:
    return float(np.mean(collapsed_flags))