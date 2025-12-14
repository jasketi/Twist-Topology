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

    # Optional: Fig S2 (hidden signals) if input files exist
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
