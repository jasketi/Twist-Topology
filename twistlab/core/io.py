
from __future__ import annotations
import numpy as np
import pandas as pd

def resample_positions(df: pd.DataFrame,
                       dt: float,
                       id_col: str = "id",
                       t_col: str = "t",
                       x_col: str = "x",
                       y_col: str = "y",
                       method: str = "linear") -> pd.DataFrame:
    """
    Resample positions for each id onto a common, equidistant time grid.

    Parameters
    ----------
    df : DataFrame with columns [id, t, x, y]
    dt : desired step in seconds (or native units)
    method : interpolation method passed to pandas.Series.interpolate

    Returns
    -------
    DataFrame with columns [id, t, x, y] on a common grid.
    """
    required = {id_col, t_col, x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out_frames = []
    for gid, g in df.groupby(id_col, sort=False):
        g = g.sort_values(t_col)
        tmin, tmax = g[t_col].iloc[0], g[t_col].iloc[-1]
        # Build grid for this ID only (later we'll outer-join by time if needed)
        grid = np.arange(tmin, tmax + 1e-12, dt)
        gi = pd.DataFrame({t_col: grid})
        # merge and interpolate
        merged = gi.merge(g[[t_col, x_col, y_col]], on=t_col, how="left")
        for c in (x_col, y_col):
            merged[c] = merged[c].interpolate(method=method).bfill().ffill()
        merged[id_col] = gid
        out_frames.append(merged[[id_col, t_col, x_col, y_col]])
    out = pd.concat(out_frames, ignore_index=True)
    return out

def compute_velocities(df: pd.DataFrame,
                       dt: float,
                       id_col: str = "id",
                       t_col: str = "t",
                       x_col: str = "x",
                       y_col: str = "y",
                       smooth_window_s: float | None = None) -> pd.DataFrame:
    """
    Finite-difference velocities per id; optional robust median smoothing.

    Returns
    -------
    DataFrame with extra columns vx, vy.
    """
    df = df.sort_values([id_col, t_col]).copy()
    # compute finite differences within each id
    vx_list, vy_list = [], []
    for gid, g in df.groupby(id_col, sort=False):
        dx = g[x_col].diff()
        dy = g[y_col].diff()
        vx = dx / dt
        vy = dy / dt
        # pad first value
        vx.iloc[0] = vx.iloc[1] if len(vx) > 1 else 0.0
        vy.iloc[0] = vy.iloc[1] if len(vy) > 1 else 0.0
        vs = pd.DataFrame({"vx": vx, "vy": vy}, index=g.index)
        if smooth_window_s and smooth_window_s > 0:
            win = max(1, int(round(smooth_window_s / dt)))
            vs["vx"] = vs["vx"].rolling(win, center=True, min_periods=1).median()
            vs["vy"] = vs["vy"].rolling(win, center=True, min_periods=1).median()
        vx_list.append(vs["vx"])
        vy_list.append(vs["vy"])
    df["vx"] = pd.concat(vx_list).sort_index()
    df["vy"] = pd.concat(vy_list).sort_index()
    return df

def ensure_float32(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("float32")
    return df
