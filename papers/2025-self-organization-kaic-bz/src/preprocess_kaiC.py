import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', required=True, help='raw KaiC CSV/TSV file (manual download)')
    ap.add_argument('--outdir', default='papers/2025-self-organization-kaic-bz/data/processed/kaiC')
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Try CSV first, fallback to TSV.
    try:
        df = pd.read_csv(infile)
    except Exception:
        df = pd.read_csv(infile, sep='\t')

    # TODO: map columns to canonical schema: time, condition, replicate, signal
    # For now: write back as parquet for fast downstream work.
    (outdir / 'kaiC_raw_as_read.parquet').write_bytes(df.to_parquet(index=False))
    print('OK: wrote', outdir / 'kaiC_raw_as_read.parquet')

if __name__ == '__main__':
    main()
