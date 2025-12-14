#!/usr/bin/env python3
"""Reproduce Fig S2 (hidden signals): PLI/drift matrices + summary tables.

Inputs (expected under data/processed/bz/hidden):
- bz_hidden_PLI_matrix.csv
- bz_hidden_drift_matrix.csv
- bz_hidden_all_pairs.csv
- bz_hidden_signal_periods.csv
- figS2_hidden_signals_summary.csv
- figS2_hidden_pairs.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_yaml(path: str) -> dict:
    import yaml  # type: ignore
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def read_matrix_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If first column looks like an index column, use it.
    if df.columns[0].startswith("Unnamed") or df.columns[0].lower() in ("id", "idx", "name", "signal", "label"):
        df = df.set_index(df.columns[0])
    return df


def save_heatmap(df: pd.DataFrame, title: str, outpath: Path):
    arr = df.values.astype(float)
    fig = plt.figure()
    plt.imshow(arr, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(df.shape[1]), list(df.columns), rotation=90)
    plt.yticks(range(df.shape[0]), list(df.index))
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--indir", default="papers/2025-self-organization-kaic-bz/data/processed/bz/hidden")
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/results")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    thr = cfg.get("thresholds", {})
    pli_thr = float(thr.get("pli_lock", 0.8))
    dw_thr = float(thr.get("dw_dt_abs_max", 1.0e-4))

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pli_df = read_matrix_csv(indir / "bz_hidden_PLI_matrix.csv")
    drift_df = read_matrix_csv(indir / "bz_hidden_drift_matrix.csv")

    save_heatmap(pli_df, f"Hidden signals: PLI matrix (thr={pli_thr})", outdir / "figS2_hidden_PLI_matrix.png")
    save_heatmap(drift_df, "Hidden signals: drift matrix", outdir / "figS2_hidden_drift_matrix.png")

    pairs = pd.read_csv(indir / "bz_hidden_all_pairs.csv")
    cols = {c.lower(): c for c in pairs.columns}
    pli_col = cols.get("pli") or cols.get("phase_locking_index") or cols.get("pli_mean")
    drift_col = cols.get("drift") or cols.get("drift_rate") or cols.get("drift_w_per_s") or cols.get("drift_rate_w_per_hr")

    if pli_col is None or drift_col is None:
        num = pairs.select_dtypes(include=[np.number]).columns.tolist()
        if len(num) >= 2:
            pli_col, drift_col = num[0], num[1]
        else:
            raise RuntimeError("Could not infer PLI/drift columns in bz_hidden_all_pairs.csv")

    drift_abs = pairs[drift_col].abs()
    locks = (pairs[pli_col] >= pli_thr) & (drift_abs <= dw_thr)

    pairs_out = pairs.copy()
    pairs_out["lock"] = locks.astype(int)
    pairs_out.to_csv(outdir / "figS2_hidden_pairs_classified.csv", index=False)

    summary = {
        "n_pairs": int(len(pairs_out)),
        "n_locks": int(locks.sum()),
        "pli_thr": pli_thr,
        "dw_dt_abs_max": dw_thr,
        "pli_col": pli_col,
        "drift_col": drift_col,
    }
    (outdir / "figS2_hidden_summary.json").write_text(json.dumps(summary, indent=2))

    top = pairs_out.sort_values(by=[pli_col], ascending=False).head(20)
    top.to_csv(outdir / "figS2_hidden_top20_by_PLI.csv", index=False)

    for fn in ["bz_hidden_signal_periods.csv", "figS2_hidden_signals_summary.csv", "figS2_hidden_pairs.csv"]:
        p = indir / fn
        if p.exists():
            (outdir / fn).write_text(p.read_text())

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": args.config,
        "indir": str(indir),
        "outdir": str(outdir),
    }
    (outdir / "run_meta_figS2_hidden.json").write_text(json.dumps(meta, indent=2))

    print("OK: wrote figS2_hidden_PLI_matrix.png")
    print("OK: wrote figS2_hidden_drift_matrix.png")
    print("OK: wrote figS2_hidden_pairs_classified.csv")


if __name__ == "__main__":
    main()
