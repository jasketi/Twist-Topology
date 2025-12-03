
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from ..core.stats import one_sided_p_greater, contiguous_intervals
from ..core.surrogates import circular_shift, block_shuffle

def _alignment(vx_i: np.ndarray, vy_i: np.ndarray, vx_j: np.ndarray, vy_j: np.ndarray) -> np.ndarray:
    num = vx_i*vx_j + vy_i*vy_j
    den = np.sqrt(vx_i**2 + vy_i**2) * np.sqrt(vx_j**2 + vy_j**2) + 1e-12
    return np.clip(num / den, -1.0, 1.0)

def sigma_lock_for_pair(t: np.ndarray,
                        vx_i: np.ndarray, vy_i: np.ndarray,
                        vx_j: np.ndarray, vy_j: np.ndarray,
                        w_s: float = 5.0, dt: float = 1.0,
                        theta_lock: float = 0.2, min_dur_s: float = 30.0,
                        Nsurr: int = 100, surrogate: str = "circular",
                        block: int = 10,
                        rng=None) -> Dict[str, Any]:
    """
    Compute σ-Lock episodes and significance for a single dyad (i,j).

    Returns dict with keys:
      P_obs, p_lock, episodes: List[(t_start,t_end)], sigma_norm: np.ndarray
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = float(dt)
    sig = _alignment(vx_i, vy_i, vx_j, vy_j)
    # robust median smoothing (window in samples)
    win = max(1, int(round(w_s / dt)))
    sig_norm = pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()
    lock_mask = sig_norm > float(theta_lock)
    minlen = max(1, int(round(min_dur_s / dt)))
    intervals = [(a,b,l) for (a,b,l) in contiguous_intervals(lock_mask, t) if l >= minlen]
    P_obs = np.sum([l for (_,_,l) in intervals]) / sig_norm.size

    # surrogates
    P_surr = []
    arr = np.column_stack([vx_i, vy_i, vx_j, vy_j])
    for s in range(Nsurr):
        if surrogate == "circular":
            arr_s = circular_shift(arr, rng=rng)
        elif surrogate == "block":
            arr_s = block_shuffle(arr, block=block, rng=rng)
        else:
            arr_s = circular_shift(arr, rng=rng)
        vi = arr_s[:, :2]; vj = arr_s[:, 2:]
        sig_s = _alignment(vi[:,0], vi[:,1], vj[:,0], vj[:,1])
        sig_s = pd.Series(sig_s).rolling(win, center=True, min_periods=1).median().to_numpy()
        mask_s = sig_s > float(theta_lock)
        intervals_s = [(a,b,l) for (a,b,l) in contiguous_intervals(mask_s, t) if l >= minlen]
        P_surr.append(np.sum([l for (_,_,l) in intervals_s]) / sig_s.size)
    p_lock = one_sided_p_greater(P_obs, np.asarray(P_surr))
    episodes = [(a,b) for (a,b,l) in intervals]
    return {
        "P_obs": float(P_obs),
        "p_lock": float(p_lock),
        "episodes": episodes,
        "sigma_norm": sig_norm,
    }

def sigma_lock(df_vel: pd.DataFrame,
               id_col: str = "id", t_col: str = "t",
               vx_col: str = "vx", vy_col: str = "vy",
               w_s: float = 5.0, theta_lock: float = 0.2, min_dur_s: float = 30.0,
               Nsurr: int = 100, surrogate: str = "circular", block: int = 10,
               pairs: List[Tuple[Any,Any]] | None = None,
               dt: float | None = None,
               rng=None) -> pd.DataFrame:
    """
    Run σ-Lock for all dyads in df_vel.

    Returns
    -------
    DataFrame with columns:
      i,j,P_obs,p_lock,episodes (list of (t_start,t_end))
    """
    rng = np.random.default_rng() if rng is None else rng
    # time step
    if dt is None:
        ts = df_vel[t_col].drop_duplicates().sort_values().to_numpy()
        dt = float(np.median(np.diff(ts)))
    ids = df_vel[id_col].drop_duplicates().to_list()
    if pairs is None:
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i+1, len(ids))]
    results = []
    for (i, j) in pairs:
        gi = df_vel[df_vel[id_col]==i].sort_values(t_col)
        gj = df_vel[df_vel[id_col]==j].sort_values(t_col)
        # align by time intersection
        merged = gi[[t_col,"vx","vy"]].merge(gj[[t_col,"vx","vy"]], on=t_col, suffixes=("_i","_j"))
        if merged.empty: 
            continue
        out = sigma_lock_for_pair(merged[t_col].to_numpy(),
                                  merged["vx_i"].to_numpy(), merged["vy_i"].to_numpy(),
                                  merged["vx_j"].to_numpy(), merged["vy_j"].to_numpy(),
                                  w_s=w_s, dt=dt, theta_lock=theta_lock, min_dur_s=min_dur_s,
                                  Nsurr=Nsurr, surrogate=surrogate, block=block, rng=rng)
        results.append({
            "i": i, "j": j,
            "P_obs": out["P_obs"],
            "p_lock": out["p_lock"],
            "episodes": out["episodes"]
        })
    return pd.DataFrame(results)
