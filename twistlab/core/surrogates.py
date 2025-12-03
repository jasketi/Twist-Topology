
from __future__ import annotations
import numpy as np

def circular_shift(x: np.ndarray, k: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    "Circularly shift a 1D or 2D array along axis 0 by k (random if None)."
    rng = np.random.default_rng() if rng is None else rng
    n = x.shape[0]
    if k is None:
        k = int(rng.integers(1, max(2, n-1)))
    return np.roll(x, shift=k, axis=0)

def phase_randomize(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Fourier phase randomization (single pass). Preserves amplitude spectrum.

    x: array of shape (T,) or (T, D). Each column randomized independently.
    Returns array of same shape.
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x)
    if x.ndim == 1:
        return _phase_rand_1d(x, rng)
    else:
        cols = []
        for d in range(x.shape[1]):
            cols.append(_phase_rand_1d(x[:, d], rng))
        return np.column_stack(cols)

def _phase_rand_1d(x: np.ndarray, rng) -> np.ndarray:
    n = x.shape[0]
    Xf = np.fft.rfft(x)
    # random phases for positive freqs excluding DC and Nyquist
    k = Xf.shape[0]
    phases = rng.random(k) * 2*np.pi
    phases[0] = 0.0
    if k > 1:
        phases[-1] = 0.0  # Nyquist no phase
    Xf_rand = np.abs(Xf) * np.exp(1j*phases)
    xr = np.fft.irfft(Xf_rand, n=n)
    # match mean/variance
    xr = (xr - xr.mean()) * (np.std(x, ddof=0) / (np.std(xr, ddof=0) + 1e-12)) + x.mean()
    return xr

def block_shuffle(x: np.ndarray, block: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Block-shuffle surrogate: keeps local autocorrelation within blocks, shuffles block order.
    Works for 1D or 2D (applied along axis 0).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x)
    n = x.shape[0]
    block = max(1, int(block))
    nb = int(np.ceil(n / block))
    idx = np.arange(nb)
    rng.shuffle(idx)
    segs = []
    for i in idx:
        s = slice(i*block, min((i+1)*block, n))
        segs.append(x[s])
    return np.concatenate(segs, axis=0)
