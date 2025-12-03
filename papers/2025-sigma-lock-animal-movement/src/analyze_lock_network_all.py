#!/usr/bin/env python3
import pandas as pd
import numpy as np

LOCKS_FILE = "locks_storks_all.csv"

print(f"1) Lese {LOCKS_FILE} ...")
locks = pd.read_csv(LOCKS_FILE)
print("   Zeilen:", len(locks))
print("   Spalten:", list(locks.columns))

# Wir gehen von diesen Spalten aus:
needed = {"i","j","P_obs","p_lock"}
if not needed.issubset(locks.columns):
    raise SystemExit(f"Erwarte mind. Spalten {needed} in {LOCKS_FILE}")

# Lock-Gewicht: beobachteter Lock-Anteil * (1 - p_lock)
locks["w"] = locks["P_obs"] * (1.0 - locks["p_lock"])

# IDs einsammeln
ids = sorted(set(locks["i"]) | set(locks["j"]))
idx = {id_: k for k, id_ in enumerate(ids)}
n = len(ids)
W = np.zeros((n, n), dtype=float)

for _, row in locks.iterrows():
    a = idx[row["i"]]
    b = idx[row["j"]]
    w = row["w"]
    W[a, b] = W[b, a] = w

# Lock-Stärke pro Individuum = Summe der Gewichte über alle Partner
lock_strength = W.sum(axis=1)

summary = (
    pd.DataFrame({
        "stork": ids,
        "lock_strength": lock_strength,
    })
    .sort_values("lock_strength", ascending=False)
)

print("\n2) Lock-Stärke pro Storch (Top 20):\n")
print(summary.head(20).to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

summary.to_csv("lock_strength_per_stork.csv", index=False)
print("\n3) FERTIG: lock_strength_per_stork.csv geschrieben.")
