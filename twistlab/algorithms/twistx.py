
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from ..core.surrogates import phase_randomize
from ..core.stats import one_sided_p_greater

def unwrap_phase(phi: np.ndarray) -> np.ndarray:
    return np.unwrap(phi)

def half_twist_indices(theta: np.ndarray) -> np.ndarray:
    """
    Given cumulative rotation Theta(t), return indices where it crosses odd multiples of π.
    """
    # compute k(t) = floor((theta/pi + 1)/2) to detect odd crossings
    k = np.floor((theta/np.pi + 1.0)/2.0).astype(int)
    dk = np.diff(k)
    # a change in k indicates we've crossed an odd multiple of π
    idx = np.where(np.abs(dk) >= 1)[0] + 1
    return idx

def compute_twistx_for_pair(t: np.ndarray,
                            dvx: np.ndarray, dvy: np.ndarray,
                            eps: float = 0.2, dt: float = 1.0,
                            Nsurr: int = 300,
                            rng=None) -> Dict[str, Any]:
    """
    TwistX (dyad) for a single pair. Returns Stopo, ptwist, event indices and times.
    """
    rng = np.random.default_rng() if rng is None else rng
    # angle and cumulative rotation
    phi = np.arctan2(dvy, dvx)
    theta = unwrap_phase(phi)
    # event indices for half-twists
    idx = half_twist_indices(theta)
    events_t = t[idx] if idx.size>0 else np.array([], dtype=float)
    # compute stability score: proximity to lagged state after last half-twist
    x = np.column_stack([dvx, dvy])
    T = x.shape[0]
    if idx.size == 0:
        Stopo = 0.0
    else:
        last_idx = -np.ones(T, dtype=int)
        # fill array with index of last event up to each time
        e_iter = iter(idx)
        e = next(e_iter, None)
        for k in range(T):
            while e is not None and e <= k:
                last_seen = e
                e = next(e_iter, None)
            if 'last_seen' in locals():
                last_idx[k] = last_seen
        # points with a defined last event and lag > 0
        valid = last_idx >= 0
        # form distances ||x(t) - x(t - tau*)|| with tau* = k - last_idx[k]
        dst = np.full(T, np.inf)
        for k in range(T):
            j = last_idx[k]
            if j >= 0 and (k-j) > 0:
                dst[k] = np.linalg.norm(x[k] - x[k-(k-j)])
        Stopo = float(np.mean(dst[valid] < float(eps))) if np.any(valid) else 0.0

    # surrogates via phase randomization of dv
    S_surr = []
    for s in range(Nsurr):
        dv_s = phase_randomize(x, rng=rng)
        phi_s = np.arctan2(dv_s[:,1], dv_s[:,0])
        theta_s = unwrap_phase(phi_s)
        idx_s = half_twist_indices(theta_s)
        if idx_s.size == 0:
            S_surr.append(0.0)
            continue
        x_s = dv_s
        T_s = x_s.shape[0]
        last_idx_s = -np.ones(T_s, dtype=int)
        e_iter = iter(idx_s); e = next(e_iter, None)
        for k in range(T_s):
            while e is not None and e <= k:
                last_seen = e
                e = next(e_iter, None)
            if 'last_seen' in locals():
                last_idx_s[k] = last_seen
        valid_s = last_idx_s >= 0
        dst_s = np.full(T_s, np.inf)
        for k in range(T_s):
            j = last_idx_s[k]
            if j >= 0 and (k-j) > 0:
                dst_s[k] = np.linalg.norm(x_s[k] - x_s[k-(k-j)])
        S_surr.append(float(np.mean(dst_s[valid_s] < float(eps))) if np.any(valid_s) else 0.0)
    ptwist = one_sided_p_greater(Stopo, np.asarray(S_surr))
    return {"Stopo": Stopo, "ptwist": float(ptwist), "events_t": events_t, "events_idx": idx}

def compute_twistx(df_vel: pd.DataFrame,
                   lock_windows: List[Tuple[float, float]] | None = None,
                   id_i=None, id_j=None,
                   id_col: str = "id", t_col: str = "t",
                   vx_col: str = "vx", vy_col: str = "vy",
                   eps: float = 0.2, Nsurr: int = 300) -> Dict[str, Any]:
    """
    Compute TwistX for a specified dyad (id_i, id_j). If lock_windows is provided,
    restrict analysis to their union.
    """
    if id_i is None or id_j is None:
        raise ValueError("id_i and id_j must be provided")
    gi = df_vel[df_vel[id_col]==id_i][[t_col, vx_col, vy_col]].sort_values(t_col)
    gj = df_vel[df_vel[id_col]==id_j][[t_col, vx_col, vy_col]].sort_values(t_col)
    merged = gi.merge(gj, on=t_col, suffixes=("_i","_j"))
    if merged.empty:
        return {"Stopo": 0.0, "ptwist": 1.0, "events_t": np.array([]), "events_idx": np.array([])}
    if lock_windows:
        # mask times inside any lock window
        t = merged[t_col].to_numpy()
        mask = np.zeros_like(t, dtype=bool)
        for a,b in lock_windows:
            mask |= (t>=a) & (t<=b)
        merged = merged[mask]
    t = merged[t_col].to_numpy()
    dvx = (merged["vx_j"] - merged["vx_i"]).to_numpy()
    dvy = (merged["vy_j"] - merged["vy_i"]).to_numpy()
    # estimate dt from t
    if len(t) < 3:
        return {"Stopo": 0.0, "ptwist": 1.0, "events_t": np.array([]), "events_idx": np.array([])}
    dt = float(np.median(np.diff(t)))
    return compute_twistx_for_pair(t, dvx, dvy, eps=eps, dt=dt, Nsurr=Nsurr)
