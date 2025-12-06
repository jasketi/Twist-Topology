#!/usr/bin/env python
"""
Wrapper script for the origin-of-life paper.

This script calls the local algorithm implementation
in origin_life_algorithm.py and reports where the results
were written.
"""

from pathlib import Path

from origin_life_algorithm import run_origin_life_example, EnvironmentParams


def main():
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "data"

    env = EnvironmentParams()  # default parameters; adjust if desired

    df = run_origin_life_example(output_dir=data_path, env=env)
    out_file = data_path / "minimal_sigma_lock_results.csv"

    print(f"Origin-of-life example finished.")
    print(f"- rows in result: {len(df)}")
    print(f"- CSV written to: {out_file}")


if __name__ == "__main__":
    main()
