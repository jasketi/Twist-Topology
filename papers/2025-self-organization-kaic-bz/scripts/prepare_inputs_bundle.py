#!/usr/bin/env python3
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone

REQUIRED = [
    "data/processed/kaiC/kaiC_summary_metrics.csv",
    "data/processed/kaiC/kaiC_kuramoto_fits.csv",
    "data/processed/kaiC/kaiC_newdata_pairwise_phase_metrics.csv",
    "data/processed/kaiC/kaiC_newdata_summary_metrics.csv",
    "data/processed/bz/fig1/bz_fig1_droplets_seconds_summary.csv",
    "data/processed/bz/hidden/bz_hidden_PLI_matrix.csv",
    "data/processed/bz/hidden/bz_hidden_drift_matrix.csv",
    "data/processed/bz/hidden/bz_hidden_all_pairs.csv",
    "data/processed/bz/hidden/bz_hidden_signal_periods.csv",
    "data/processed/bz/hidden/figS2_hidden_pairs.csv",
    "data/processed/bz/hidden/figS2_hidden_signals_summary.csv",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_dir", default="papers/2025-self-organization-kaic-bz")
    ap.add_argument("--outdir", default="papers/2025-self-organization-kaic-bz/inputs_bundle")
    ap.add_argument("--zipname", default="kaic-bz_processed_inputs.zip")
    args = ap.parse_args()

    paper_dir = Path(args.paper_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    staging = outdir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    missing, included = [], []
    for rel in REQUIRED:
        src = paper_dir / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        included.append(rel)

    manifest = {
        "paper": "2025-self-organization-kaic-bz",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "included": [],
        "missing": missing,
        "notes": "Processed inputs bundle for reproducing Table 1 (KaiC), Fig 1 (BZ), Fig 2 (KaiC), Fig S2 (BZ hidden).",
    }
    for rel in included:
        p = staging / rel
        manifest["included"].append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})

    (outdir / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2))

    zip_path = outdir / args.zipname
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(outdir / "inputs_manifest.json", arcname="inputs_manifest.json")
        for rel in included:
            z.write(staging / rel, arcname=rel)

    shutil.rmtree(staging)

    print("Wrote:", zip_path)
    print("Wrote:", outdir / "inputs_manifest.json")
    if missing:
        print("WARNING missing files (bundle still created):")
        for m in missing:
            print(" -", m)

if __name__ == "__main__":
    main()
