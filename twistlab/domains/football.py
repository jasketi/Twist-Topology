
from __future__ import annotations
import numpy as np
import pandas as pd

def align_events_to_windows(events_df: pd.DataFrame, t_col: str="t",
                            windows: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    """
    Mark events that occur inside any given window list.
    """
    if windows is None:
        df = events_df.copy()
        df["inside_window"] = True
        return df
    mask = []
    t = events_df[t_col].to_numpy()
    for tt in t:
        m = any(a <= tt <= b for (a,b) in windows)
        mask.append(m)
    out = events_df.copy()
    out["inside_window"] = mask
    return out

def structural_performance_index(twist_density_window: float,
                                 twist_trend: float,
                                 ball_loss_rate: float,
                                 alpha: float=1.0, beta: float=1.0, gamma: float=1.0) -> float:
    """
    SPI = alpha * TwistX[-5,0] + beta * DeltaTwistX[-10,0->-5,0] - gamma * BallLossRate
    """
    return alpha*twist_density_window + beta*twist_trend - gamma*ball_loss_rate
