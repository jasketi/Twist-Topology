#!/usr/bin/env python3
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

def load_yaml(path: str) -> dict:
    import yaml  # type: ignore
    with open(path,"r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--infile", default="papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_newdata_pairwise_phase_metrics.csv")
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/results")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    thr = cfg.get("thresholds", {})
    pli_thr = float(thr.get("pli_lock", 0.8))
    drift_thr = float(thr.get("dw_dt_abs_max", 1.0e-4))

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    # Column inference (robust)
    cols = {c.lower(): c for c in df.columns}
    pair_col = cols.get("pair") or cols.get("label") or cols.get("pair_label") or df.columns[0]
    pli_col = cols.get("pli") or cols.get("phase_locking_index")
    drift_col = cols.get("drift_rate_w_per_hr") or cols.get("drift") or cols.get("drift_rate")

    if pli_col is None or drift_col is None:
        raise RuntimeError(f"Could not find required columns. Have columns: {list(df.columns)}")

    df_out = df[[pair_col, drift_col, pli_col]].copy()
    df_out = df_out.rename(columns={pair_col:"pair", drift_col:"drift_rate_W_per_hr", pli_col:"PLI"})
    df_out["abs_drift_rate_W_per_hr"] = df_out["drift_rate_W_per_hr"].abs()
    df_out["lock"] = ((df_out["PLI"] >= pli_thr) & (df_out["abs_drift_rate_W_per_hr"] <= drift_thr)).astype(int)

    # Summary row
    summary = {
        "n_pairs": int(len(df_out)),
        "n_locks": int(df_out["lock"].sum()),
        "pli_thr": pli_thr,
        "drift_thr_W_per_hr": drift_thr,
    }

    # Write CSV
    csv_path = outdir / "table1_kaiC.csv"
    df_out.to_csv(csv_path, index=False)

    # LaTeX rows (simple)
    tex_path = outdir / "table1_kaiC_rows.tex"
    lines = []
    for _, r in df_out.iterrows():
        lines.append(f"{r['pair']} & {r['drift_rate_W_per_hr']:.6g} & {r['PLI']:.3f} & {int(r['lock'])} \\\\")
    tex_path.write_text("\n".join(lines) + "\n")

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "config": args.config,
        "infile": str(infile),
        "summary": summary,
    }
    (outdir / "run_meta_table1_kaiC.json").write_text(json.dumps(meta, indent=2))

    print("Wrote:", csv_path)
    print("Wrote:", tex_path)
    print("Wrote:", outdir / "run_meta_table1_kaiC.json")

if __name__ == "__main__":
    main()
