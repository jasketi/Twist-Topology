"""Core metrics for the kaic-bz paper.

Definitions (paper):
- W = Δφ / (2π)
- PLI = |⟨exp(i(φ_i - φ_j))⟩|
- dW/dt = (1/2π) d(φ_i - φ_j)/dt

Inputs are phase time series φ(t) in radians.
"""

from __future__ import annotations

import numpy as np

def winding_number(phi_i: np.ndarray, phi_j: np.ndarray) -> float:
    """Return W over the full interval using end-to-end phase difference."""
    dphi = (phi_i[-1] - phi_j[-1]) - (phi_i[0] - phi_j[0])
    return float(dphi / (2.0 * np.pi))

def pli(phi_i: np.ndarray, phi_j: np.ndarray) -> float:
    """Phase-locking index over the full interval."""
    z = np.exp(1j * (phi_i - phi_j))
    return float(np.abs(np.mean(z)))

def dw_dt(phi_i: np.ndarray, phi_j: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Instantaneous dW/dt time series (same length as t)."""
    dphi = (phi_i - phi_j)
    # numerical derivative w.r.t. t
    ddphi_dt = np.gradient(dphi, t)
    return ddphi_dt / (2.0 * np.pi)

def windowed_metrics(phi_i: np.ndarray, phi_j: np.ndarray, t: np.ndarray, window: int, step: int) -> dict:
    """Compute windowed PLI and median(|dW/dt|) over sliding windows."""
    plis = []
    dw_abs_med = []
    centers = []
    for start in range(0, len(t) - window + 1, step):
        end = start + window
        plis.append(pli(phi_i[start:end], phi_j[start:end]))
        dwdt = dw_dt(phi_i[start:end], phi_j[start:end], t[start:end])
        dw_abs_med.append(float(np.median(np.abs(dwdt))))
        centers.append(float(0.5 * (t[start] + t[end-1])))
    return {'t_center': np.array(centers), 'pli': np.array(plis), 'dw_abs_median': np.array(dw_abs_med)}
