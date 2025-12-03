
from __future__ import annotations
import numpy as np
import pandas as pd

def one_sided_p_greater(obs: float, surrogates: np.ndarray) -> float:
    surrogates = np.asarray(surrogates)
    return (np.sum(surrogates >= obs) + 1.0) / (surrogates.size + 1.0)

def two_sided_p(obs: float, surrogates: np.ndarray) -> float:
    surrogates = np.asarray(surrogates)
    return (np.sum(np.abs(surrogates) >= np.abs(obs)) + 1.0) / (surrogates.size + 1.0)

def contiguous_intervals(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float, int]]:
    """
    Extract contiguous intervals where mask==True.

    Returns list of (t_start, t_end, length_in_samples)
    """
    mask = np.asarray(mask, dtype=bool)
    t = np.asarray(t)
    out = []
    if mask.size == 0:
        return out
    on = False
    start_idx = 0
    for i, m in enumerate(mask):
        if m and not on:
            on = True
            start_idx = i
        elif not m and on:
            on = False
            out.append((t[start_idx], t[i-1], i-start_idx))
    if on:
        out.append((t[start_idx], t[-1], mask.size - start_idx))
    return out
