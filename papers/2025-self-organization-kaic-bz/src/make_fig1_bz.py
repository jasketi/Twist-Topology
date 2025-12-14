#!/usr/bin/env python3
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_yaml(path: str) -> dict:
    import yaml  # type: ignore
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def infer_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    # fallback: contains-match
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def hist_plot(series, outpath, title, xlabel):
    series = pd.to_numeric(series, errors="coerce").dropna()
    fig = plt.figure()
    plt.hist(series.values, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--infile", default="papers/2025-self-organization-kaic-bz/data/processed/bz/fig1/bz_fig1_droplets_seconds_summary.csv")
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/results")
    args = ap.parse_args()

    _ = load_yaml(args.config)  # reserved for thresholds later if needed

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    # Try infer columns
    winding_col = infer_col(df, ["winding", "w_total_cycles", "w_total", "winding_total", "winding_total_cycles"])
    period_col  = infer_col(df, ["period_s", "dom_period_s", "dominant_period_s", "dom_period", "period"])

    # Always write a quick preview table for reproducibility/debug
    (outdir / "fig1_bz_input_head.csv").write_text(df.head(20).to_csv(index=False))

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "config": args.config,
        "infile": str(infile),
        "columns": list(df.columns),
        "winding_col": winding_col,
        "period_col": period_col,
        "n_rows": int(len(df)),
    }
    (outdir / "run_meta_fig1_bz.json").write_text(json.dumps(meta, indent=2))

    if winding_col is None and period_col is None:
        raise RuntimeError(f"Could not infer winding/period columns. See {outdir/'fig1_bz_input_head.csv'} and meta json.")

    if winding_col is not None:
        hist_plot(df[winding_col], outdir / "fig1_bz_winding_hist.png",
                  "BZ Fig1: winding distribution", winding_col)

    if period_col is not None:
        hist_plot(df[period_col], outdir / "fig1_bz_period_hist.png",
                  "BZ Fig1: dominant period distribution", period_col)

    print("OK: wrote", outdir / "run_meta_fig1_bz.json")
    print("OK: wrote", outdir / "fig1_bz_input_head.csv")
    if winding_col: print("OK: wrote fig1_bz_winding_hist.png")
    if period_col:  print("OK: wrote fig1_bz_period_hist.png")

if __name__ == "__main__":
    main()
