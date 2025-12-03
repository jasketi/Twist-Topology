#!/usr/bin/env python3
import pandas as pd

TRIADS_FILE = "triads_scan.csv"

print(f"1) Lese {TRIADS_FILE} ...")
df = pd.read_csv(TRIADS_FILE)
print("   Zeilen:", len(df))
print("   Spalten:", list(df.columns))

rows = []
for _, row in df.iterrows():
    for role in ["A","B","C"]:
        rows.append({
            "stork": row[role],
            "Striad": row["Striad"],
            "ptriad": row["ptriad"],
            "Ncycles": row["Ncycles"],
        })

long = pd.DataFrame(rows)
summary = (
    long
    .groupby("stork")
    .agg(
        n_triads=("Striad","count"),
        mean_Striad=("Striad","mean"),
        max_Striad=("Striad","max"),
        sum_Ncycles=("Ncycles","sum"),
    )
    .sort_values("mean_Striad", ascending=False)
)

print("\n2) Triaden-Rollen pro Storch (sortiert nach mittlerem Striad):\n")
print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))

summary.to_csv("triad_roles_per_stork.csv")
print("\n3) FERTIG: triad_roles_per_stork.csv geschrieben.")
