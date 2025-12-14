#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="papers/2025-self-organization-kaic-bz")
    ap.add_argument("--manifest", default="papers/2025-self-organization-kaic-bz/inputs_bundle/inputs_manifest.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve()
    man = json.loads(manifest_path.read_text())

    ok = True
    for item in man.get("included", []):
        rel = item["path"]
        expected = item["sha256"]
        p = root / rel
        if not p.exists():
            print("MISSING:", rel)
            ok = False
            continue
        got = sha256(p)
        if got != expected:
            print("HASH MISMATCH:", rel)
            print(" expected:", expected)
            print(" got     :", got)
            ok = False

    if ok:
        print("OK: all included files exist and match SHA256.")
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
