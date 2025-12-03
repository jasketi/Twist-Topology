#!/usr/bin/env python3
import pandas as pd
import numpy as np

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.algorithms.sigma_lock import sigma_lock

print("1) Lese storks_positions.csv ...")
df = pd.read_csv("storks_positions.csv")
print("   Spalten:", list(df.columns))
print("   Anzahl Zeilen:", len(df))

# Falls noch keine metrischen Koordinaten vorhanden sind: lat/lon -> x,y
if {"lat","lon"}.issubset(df.columns) and {"x","y"}.isdisjoint(df.columns):
    print("2) Wandle lat/lon -> x,y um ...")
    from twistlab.domains.movebank import latlon_to_xy
    x, y = latlon_to_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.copy()
    df["x"] = x
    df["y"] = y
    print("   Fertig mit lat/lon -> x,y.")

print("3) Resampling auf dt = 1.0 s ...")
df_r = resample_positions(df[["id","t","x","y"]], dt=1.0)
print("   Resampled shape:", df_r.shape)
print("   IDs nach Resampling:", df_r['id'].nunique())

print("4) Geschwindigkeiten berechnen (smooth = 15.0 s) ...")
df_v = compute_velocities(df_r, dt=1.0, smooth_window_s=15.0)
print("   Velocities shape:", df_v.shape)

# Für den Test nur die ersten paar Störche nehmen, damit es schneller geht
ids = df_v["id"].drop_duplicates().to_list()
print("5) Verfügbare IDs (erste 10):", ids[:10])
max_ids = min(5, len(ids))
subset_ids = ids[:max_ids]
pairs = [(subset_ids[i], subset_ids[j]) for i in range(max_ids) for j in range(i+1, max_ids)]
print("   Nutze nur diese IDs:", subset_ids)
print("   Dyaden-Paare für Test:", pairs)

print("6) σ-Locks für diese Paare berechnen (window=25.0, theta=0.2, minlen=150.0, Nsurr=20) ...")
locks = sigma_lock(
    df_v,
    w_s=25.0,
    theta_lock=0.2,
    min_dur_s=150.0,
    Nsurr=20,
    surrogate="circular",
    pairs=pairs,
    dt=1.0,
)
print("   Ergebnis: ", len(locks), "Dyaden")

print("7) Schreibe Ergebnis-Datei locks_storks_subset.csv ...")
locks.to_csv("locks_storks_subset.csv", index=False)
print("FERTIG: locks_storks_subset.csv geschrieben.")
