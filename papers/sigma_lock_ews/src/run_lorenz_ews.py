"""
Paper reproduction entry point (sigma-lock-ews).
Creates run metadata and placeholder outputs.

Usage (from repo root):
  python -m papers.sigma_lock_ews.src.run_lorenz_ews --config papers/sigma-lock-ews/configs/quick.yaml
"""
import argparse, json, os, subprocess
from datetime import datetime

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="papers/sigma-lock-ews/results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "config_path": args.config,
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # TODO: Wire to actual twistlab implementation and generate figures/tables.
    with open(os.path.join(args.outdir, "PLACEHOLDER.txt"), "w") as f:
        f.write("TODO: Connect to twistlab and reproduce paper figures/tables.\n")

    print("OK: wrote outputs to", args.outdir)

if __name__ == "__main__":
    main()
