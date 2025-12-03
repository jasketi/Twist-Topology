
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

def damped_cosine(t, A, omega, tau0, c0):
    return A * np.cos(omega*t) * np.exp(-t/float(tau0)) + c0

def fit_damped_cosine_correlation(t: np.ndarray, C: np.ndarray, p0=None, bounds=(-np.inf, np.inf)):
    """
    Fit C(t) with A cos(omega t) exp(-t/tau0) + c0.
    Returns params dict and covariance diag.
    """
    t = np.asarray(t); C = np.asarray(C)
    if p0 is None:
        # naive initial guesses
        A0 = (C.max()-C.min())/2.0
        omega0 = 2*np.pi/(t[-1]-t[0] + 1e-9)
        tau0 = (t[-1]-t[0])/2.0
        c0 = C.mean()
        p0 = [A0, omega0, tau0, c0]
    popt, pcov = curve_fit(damped_cosine, t, C, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.maximum(np.diag(pcov), 0))
    params = {"A": float(popt[0]), "omega": float(popt[1]), "tau0": float(popt[2]), "c0": float(popt[3])}
    errors = {"A": float(perr[0]), "omega": float(perr[1]), "tau0": float(perr[2]), "c0": float(perr[3])}
    return params, errors
