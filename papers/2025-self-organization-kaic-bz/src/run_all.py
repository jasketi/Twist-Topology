#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

def load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML missing. Install: pip install pyyaml") from e
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/results")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": args.config,
        "mode": cfg.get("mode", "unknown"),
        "thresholds": cfg.get("thresholds", {}),
        "windowing": cfg.get("windowing", {}),
        "data": cfg.get("data", {}),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # Minimal reproducibility spine: create a targets file + table skeleton.
    (outdir / "targets.txt").write_text("\n".join([
        "tables_skeleton.csv", "run_meta.json", "targets.txt"
    ]) + "\n")

    (outdir / "tables_skeleton.csv").write_text(
        "target,description,status\n"
        "Table1,Pairwise W/PLI/dWdt summary (to be wired),TODO\n"
        "Table2,Lock/non-lock regime counts (to be wired),TODO\n"
        "Table3,Robustness checks summary (to be wired),TODO\n"
    )

    print("OK: wrote", outdir / "run_meta.json")
    print("OK: wrote", outdir / "tables_skeleton.csv")

    # Optional: Table 1 (KaiC) if processed pairwise metrics exist
    kaiC_in = Path("papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_newdata_pairwise_phase_metrics.csv")
    if kaiC_in.exists():
        rc = os.system(f"python papers/2025-self-organization-kaic-bz/src/make_table1_kaiC.py --config {args.config} --infile {kaiC_in} --outdir {outdir}")
        if rc != 0:
            raise RuntimeError(f"Table1 KaiC step failed with exit code {rc}")

    # Optional: Fig S2 (hidden signals) if input files exist

    # Optional: Fig 2 (KaiC summary + Kuramoto fits) if processed CSVs exist
    kaiC_sum = Path("papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_summary_metrics.csv")
    kaiC_kur = Path("papers/2025-self-organization-kaic-bz/data/processed/kaiC/kaiC_kuramoto_fits.csv")
    if kaiC_sum.exists() and kaiC_kur.exists():
        rc = os.system(f"python papers/2025-self-organization-kaic-bz/src/make_fig2_kaiC.py --config {args.config} --summary_csv {kaiC_sum} --kuramoto_csv {kaiC_kur} --outdir {outdir}")
        if rc != 0:
            raise RuntimeError(f"Fig2 KaiC step failed with exit code {rc}")

    # Optional: Fig 1 (BZ droplets summary) if processed fig1 file exists
    bz_fig1_in = Path("papers/2025-self-organization-kaic-bz/data/processed/bz/fig1/bz_fig1_droplets_seconds_summary.csv")
    if bz_fig1_in.exists():
        rc = os.system(f"python papers/2025-self-organization-kaic-bz/src/make_fig1_bz.py --config {args.config} --infile {bz_fig1_in} --outdir {outdir}")
        if rc != 0:
            raise RuntimeError(f"Fig1 BZ step failed with exit code {rc}")
    hidden_dir = Path("papers/2025-self-organization-kaic-bz/data/processed/bz/hidden")
    if (hidden_dir / "bz_hidden_PLI_matrix.csv").exists():
        cmd = (
            "python papers/2025-self-organization-kaic-bz/src/make_figS2_hidden.py "
            f"--config {args.config} --indir {hidden_dir} --outdir {outdir}"
        )
        rc = os.system(cmd)
        if rc != 0:
            raise RuntimeError(f"FigS2 hidden step failed with exit code {rc}")

if __name__ == "__main__":
    main()
