#!/usr/bin/env python3
import pandas as pd
import numpy as np
from time import time

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.domains.movebank import latlon_to_xy
from twistlab.algorithms.sigma_lock import sigma_lock

DT = 1.0
SMOOTH = 15.0
WINDOW = 25.0
THETA = 0.2
MINLEN = 150.0
NSURR = 30          # etwas leichter als 100, sonst wird's brutal

print("1) Lese storks_positions.csv ...")
df = pd.read_csv("storks_positions.csv")
print("   Zeilen:", len(df))
print("   Spalten:", list(df.columns))

# lat/lon -> x,y
if {"lat","lon"}.issubset(df.columns) and {"x","y"}.isdisjoint(df.columns):
    print("2) lat/lon -> x,y ...")
    x, y = latlon_to_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.copy()
    df["x"] = x
    df["y"] = y
    print("   fertig mit Projektion.")
elif not {"x","y"}.issubset(df.columns):
    raise SystemExit("Erwarte Spalten x,y oder lat,lon in storks_positions.csv")

print(f"3) Resampling auf dt = {DT} s ...")
t0 = time()
df_r = resample_positions(df[["id","t","x","y"]], dt=DT)
print("   fertig, Shape df_r:", df_r.shape, f"(dauer {time()-t0:0.1f}s)")

print(f"4) Geschwindigkeiten berechnen (smooth = {SMOOTH} s) ...")
t1 = time()
df_v = compute_velocities(df_r, dt=DT, smooth_window_s=SMOOTH)
print("   fertig, Shape df_v:", df_v.shape, f"(dauer {time()-t1:0.1f}s)")
print("   Anzahl IDs:", df_v["id"].nunique())

print("5) σ-Locks für alle Dyaden berechnen ...")
print(f"   Parameter: window={WINDOW}, theta={THETA}, minlen={MINLEN}, Nsurr={NSURR}")
t2 = time()
locks = sigma_lock(
    df_v,
    w_s=WINDOW,
    theta_lock=THETA,
    min_dur_s=MINLEN,
    Nsurr=NSURR,
    surrogate="circular",
    dt=DT,
)
print("   sigma_lock fertig, Dauer:", f"{time()-t2:0.1f}s")
print("   Anzahl Dyaden in Ergebnis:", len(locks))

print("6) Schreiben: locks_storks_all.csv ...")
locks.to_csv("locks_storks_all.csv", index=False)
print("FERTIG: locks_storks_all.csv geschrieben.")
