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

def infer(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary_csv", default="papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_summary_metrics.csv")
    ap.add_argument("--kuramoto_csv", default="papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_kuramoto_fits.csv")
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/results")
    args = ap.parse_args()

    _ = load_yaml(args.config)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Fig2A: period vs LD ---
    dfS = pd.read_csv(args.summary_csv)
    ld_col = infer(dfS, ["daylength_h", "daylength", "ld_h", "ld"])
    # If no explicit daylength column exists, try to parse it from series labels like "daylength8"
    if ld_col is None and "series" in dfS.columns:
        dfS["daylength_h"] = (
            dfS["series"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
        )
        ld_col = "daylength_h"
    per_col = infer(dfS, ["period_acf", "period", "period_hr", "period_h"])
    if ld_col is None or per_col is None:
        (outdir / "fig2_kaiC_summary_head.csv").write_text(dfS.head(30).to_csv(index=False))
        raise RuntimeError("Could not infer daylength/period columns for kaiC_summary_metrics.csv")

    fig = plt.figure()
    plt.plot(pd.to_numeric(dfS[ld_col], errors="coerce"), pd.to_numeric(dfS[per_col], errors="coerce"), marker="o", linestyle="none")
    plt.xlabel(ld_col)
    plt.ylabel(per_col)
    plt.title("KaiC: period vs daylength")
    plt.tight_layout()
    fig.savefig(outdir / "fig2A_kaiC_period_vs_LD.png", dpi=200)
    plt.close(fig)

    # --- Fig2B: K_fit vs DeltaOmega ---
    dfK = pd.read_csv(args.kuramoto_csv)
    K_col = infer(dfK, ["k_fit", "k"])
    Dw_col = infer(dfK, ["deltaomega", "delta_omega", "domega", "deltaω", "delta_w"])
    r2_col = infer(dfK, ["r2", "r_squared"])
    if K_col is None or Dw_col is None:
        (outdir / "fig2_kaiC_kuramoto_head.csv").write_text(dfK.head(30).to_csv(index=False))
        raise RuntimeError("Could not infer K_fit / Δω columns for kaiC_kuramoto_fits.csv")

    fig = plt.figure()
    x = pd.to_numeric(dfK[Dw_col], errors="coerce")
    y = pd.to_numeric(dfK[K_col], errors="coerce")
    plt.plot(x, y, marker="o", linestyle="none")
    plt.xlabel(Dw_col)
    plt.ylabel(K_col)
    ttl = "KaiC: K_fit vs Δω"
    if r2_col is not None:
        ttl += f" (colored-by not implemented; R2 col: {r2_col})"
    plt.title(ttl)
    plt.tight_layout()
    fig.savefig(outdir / "fig2B_kaiC_K_vs_Domega.png", dpi=200)
    plt.close(fig)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "config": args.config,
        "summary_csv": args.summary_csv,
        "kuramoto_csv": args.kuramoto_csv,
        "inferred": {"ld_col": ld_col, "per_col": per_col, "K_col": K_col, "Dw_col": Dw_col, "r2_col": r2_col},
    }
    (outdir / "run_meta_fig2_kaiC.json").write_text(json.dumps(meta, indent=2))

    print("OK: wrote fig2A_kaiC_period_vs_LD.png")
    print("OK: wrote fig2B_kaiC_K_vs_Domega.png")
    print("OK: wrote run_meta_fig2_kaiC.json")

if __name__ == "__main__":
    main()
