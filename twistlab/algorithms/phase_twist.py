
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import hilbert

def phase_metrics(y_i: np.ndarray, y_j: np.ndarray, fs: float):
    """
    Compute phases via Hilbert transform and return W (winding), PLI, and dW/dt.

    Returns dict with W, PLI, drift
    """
    y_i = np.asarray(y_i); y_j = np.asarray(y_j)
    # analytic signals
    zi = hilbert(y_i); zj = hilbert(y_j)
    phi_i = np.unwrap(np.angle(zi))
    phi_j = np.unwrap(np.angle(zj))
    dphi = phi_i - phi_j
    W = (phi_i[-1] - phi_i[0])/(2*np.pi)
    PLI = np.abs(np.mean(np.exp(1j*(phi_i - phi_j))))
    drift = (dphi[-1] - dphi[0])/(2*np.pi) / (len(dphi)/fs)  # winding per second
    return {"W": float(W), "PLI": float(PLI), "dW_dt": float(drift)}
