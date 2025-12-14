#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def load_crosscorr(txt_path):
    xs, ys = [], []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                x = float(parts[0])
            except ValueError:
                continue
            if len(parts) < 2:
                continue
            try:
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x); ys.append(y)
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    idx = np.argsort(x)
    return x[idx], y[idx]

def infer_meta(fname):
    config = "cis" if "cis_" in fname else ("trans" if "trans_" in fname else "unknown")
    if "noIAA" in fname:
        cond = "noIAA"
    elif "_IAA_" in fname:
        cond = "IAA"
    elif "WT" in fname:
        cond = "WT"
    elif "noTIR" in fname:
        cond = "noTIR"
    else:
        cond = "other"
    return config, cond

def estimate_period_fft(x, y, window_frames=400):
    zero_idx = np.argmin(np.abs(x))
    half = window_frames // 2
    s = max(0, zero_idx - half)
    e = min(len(x), zero_idx + half)
    xw = x[s:e]; yw = y[s:e]
    if len(xw) < 20:
        return np.nan
    yw = yw - np.mean(yw)
    dx = np.median(np.diff(xw))
    if dx == 0:
        return np.nan
    freqs = np.fft.rfftfreq(len(yw), d=dx)
    spec = np.abs(np.fft.rfft(yw))
    if len(spec) < 3:
        return np.nan
    spec[0] = 0
    k = np.argmax(spec)
    f = freqs[k]
    return np.nan if f <= 0 else 1.0 / f

def estimate_tau_simple(x, y):
    zero_idx = np.argmin(np.abs(x))
    y0 = y - np.mean(y)
    peak = np.max(np.abs(y0))
    if peak <= 0:
        return np.nan
    thresh = peak / np.e
    for i in range(zero_idx, len(x)):
        if abs(y0[i]) < thresh:
            return abs(x[i] - x[zero_idx])
    return np.nan

def pick(files, contains):
    for fp in files:
        if contains in os.path.basename(fp):
            return fp
    return None

def main():
    files = sorted(glob.glob(str(RAW / "*cross-correlation*.txt")))
    if not files:
        raise SystemExit(f"No cross-correlation txt files found in {RAW}")

    rows = []
    for fp in files:
        fname = os.path.basename(fp)
        config, cond = infer_meta(fname)
        x, y = load_crosscorr(fp)
        T = estimate_period_fft(x, y)
        tau = estimate_tau_simple(x, y)
        rows.append(dict(file=fname, config=config, condition=cond, T_frames=T, tau_frames=tau))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fit_parameters.csv", index=False)

    # Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    exA = [
        ("GAL1-10_cis_WT_cross-correlation", "GAL1-10 cis WT", None),
        ("GAL10-7_cis_WT_cross-correlation", "GAL10-7 cis WT", None),
        ("GAL1-10_trans_WT_cross-correlation", "GAL1-10 trans WT", "--"),
    ]
    ax = axes[0]
    for key, label, ls in exA:
        fp = pick(files, key)
        if fp is None:
            continue
        x, y = load_crosscorr(fp)
        ax.plot(x, y, label=label, linestyle=ls if ls else "-")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Cross-correlation C(t)")
    ax.set_title("cis vs. trans cross-correlations")
    ax.legend()

    ax2 = axes[1]
    ax2.hist(df["T_frames"].dropna().values, bins=10)
    ax2.set_xlabel("Oscillation period T (frames)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of estimated periods")

    fig.tight_layout()
    fig.savefig(FIG / "fig_cis_trans.png", dpi=300)
    fig.savefig(FIG / "fig_cis_trans.pdf", dpi=300)

    # Figure 2 (cis only)
    cis = df[df["config"] == "cis"].copy()
    conds = ["WT", "noIAA", "IAA"]
    means_T = [cis[cis["condition"] == c]["T_frames"].mean() for c in conds]
    std_T   = [cis[cis["condition"] == c]["T_frames"].std()  for c in conds]
    means_tau = [cis[cis["condition"] == c]["tau_frames"].mean() for c in conds]
    std_tau   = [cis[cis["condition"] == c]["tau_frames"].std()  for c in conds]

    import matplotlib.gridspec as gridspec
    fig2 = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    axA = fig2.add_subplot(gs[0, 0])
    axB = fig2.add_subplot(gs[0, 1])
    axC = fig2.add_subplot(gs[1, :])

    xidx = np.arange(len(conds))
    axA.bar(xidx, means_T, yerr=std_T)
    axA.set_xticks(xidx); axA.set_xticklabels(conds)
    axA.set_ylabel("T (frames)")
    axA.set_title("A  Mean period (cis)")

    axB.bar(xidx, means_tau, yerr=std_tau)
    axB.set_xticks(xidx); axB.set_xticklabels(conds)
    axB.set_ylabel("τ₀ (frames)")
    axB.set_title("B  Mean damping (cis)")

    # Deterministic examples: use Figure2 variants for duplicated basenames
    exC = [
        ("GAL1-10_cis_WT_cross-correlation", "WT"),
        ("GAL1-10_cis_noIAA_cross-correlation_Figure2_PanelB", "noIAA"),
        ("GAL1-10_cis_IAA_cross-correlation_Figure2_PanelC", "IAA"),
    ]
    for key, label in exC:
        fp = pick(files, key)
        if fp is None:
            continue
        x, y = load_crosscorr(fp)
        axC.plot(x, y, label=label)
    axC.set_xlabel("Lag (frames)")
    axC.set_ylabel("Cross-correlation C(t)")
    axC.set_title("C  Example cis correlations under topological modulation")
    axC.legend()

    fig2.tight_layout()
    fig2.savefig(FIG / "fig_modulators.png", dpi=300)
    fig2.savefig(FIG / "fig_modulators.pdf", dpi=300)

    print("OK: wrote figures to", FIG)
    print("OK: wrote parameters to", OUT / "fit_parameters.csv")

if __name__ == "__main__":
    main()
