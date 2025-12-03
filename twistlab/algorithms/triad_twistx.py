
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from ..core.surrogates import circular_shift

def _events_from_series(times: np.ndarray, series: List[Tuple[float,float]]) -> np.ndarray:
    # Not used; events are provided directly as arrays of times.
    return times

def _count_cyclic_order(eA: np.ndarray, eB: np.ndarray, eC: np.ndarray, max_lag: float) -> int:
    """
    Count number of triadic cycles A->B->C within max_lag. Greedy matching.
    """
    cnt = 0
    i=j=k=0
    while i < len(eA) and j < len(eB) and k < len(eC):
        tA = eA[i]
        # find B after A
        while j < len(eB) and eB[j] < tA: j += 1
        if j>=len(eB): break
        if eB[j] - tA > max_lag:
            i += 1; continue
        tB = eB[j]
        # find C after B
        while k < len(eC) and eC[k] < tB: k += 1
        if k>=len(eC): break
        if eC[k] - tB > max_lag:
            j += 1; continue
        # one cycle found
        cnt += 1
        i += 1; j += 1; k += 1
    return cnt

def triad_cycles(events_AB: np.ndarray, events_BC: np.ndarray, events_CA: np.ndarray,
                 max_lag: float = 2.0, Nsurr: int = 300, rng=None) -> Dict[str, Any]:
    """
    Triad-TwistX cycles and significance.

    Parameters
    ----------
    events_XY : arrays of event times (half-twists) for each dyad
    max_lag : temporal tolerance between successive events in a cycle

    Returns
    -------
    dict with Striad, ptriad, Ncycles, counts per dyad.
    """
    rng = np.random.default_rng() if rng is None else rng
    eA, eB, eC = np.asarray(events_AB), np.asarray(events_BC), np.asarray(events_CA)
    N_AB, N_BC, N_CA = len(eA), len(eB), len(eC)
    Ncycles = _count_cyclic_order(eA, eB, eC, max_lag) + _count_cyclic_order(eA, eC, eB, max_lag)
    denom = max(1, min(N_AB, N_BC, N_CA))
    Striad = Ncycles / denom
    # surrogates via independent circular shifts of event series (implemented by time-origin shift)
    # To shift, we wrap times modulo full range
    if N_AB+N_BC+N_CA == 0:
        return {"Striad": 0.0, "ptriad": 1.0, "Ncycles": 0, "counts": (N_AB, N_BC, N_CA)}
    all_times = np.concatenate([eA, eB, eC])
    T_min, T_max = all_times.min(), all_times.max()
    T = T_max - T_min + 1e-9
    S = []
    for s in range(Nsurr):
        shifts = rng.random(3) * T
        eA_s = ((eA - T_min + shifts[0]) % T) + T_min
        eB_s = ((eB - T_min + shifts[1]) % T) + T_min
        eC_s = ((eC - T_min + shifts[2]) % T) + T_min
        n = _count_cyclic_order(np.sort(eA_s), np.sort(eB_s), np.sort(eC_s), max_lag)             + _count_cyclic_order(np.sort(eA_s), np.sort(eC_s), np.sort(eB_s), max_lag)
        S.append(n / denom)
    ptriad = (np.sum(np.asarray(S) >= Striad) + 1.0) / (len(S) + 1.0)
    return {"Striad": float(Striad), "ptriad": float(ptriad), "Ncycles": int(Ncycles), "counts": (int(N_AB), int(N_BC), int(N_CA))}
