#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json, ast

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.domains.movebank import latlon_to_xy
from twistlab.algorithms.twistx import compute_twistx
from twistlab.algorithms.triad_twistx import triad_cycles

DT = 1.0
SMOOTH = 15.0

# Drei Störche für die Triade
ID_A = "Balou / DER AN916 (eobs 3913)"
ID_B = "Bello / DER AU011 (eobs 3935)"
ID_C = "Beringa / DER AN869 (eobs 3645)"

print("1) Lese storks_positions.csv ...")
df = pd.read_csv("storks_positions.csv")
print("   Zeilen:", len(df))

# lat/lon -> x,y (metrisch)
if {"lat","lon"}.issubset(df.columns) and {"x","y"}.isdisjoint(df.columns):
    print("2) lat/lon -> x,y ...")
    x, y = latlon_to_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.copy()
    df["x"] = x
    df["y"] = y
    print("   fertig.")

print("3) Resampling auf dt = 1.0 s ...")
df_r = resample_positions(df[["id","t","x","y"]], dt=DT)
print("   Resampled shape:", df_r.shape)

print("4) Geschwindigkeiten berechnen (smooth = 15.0 s) ...")
df_v = compute_velocities(df_r, dt=DT, smooth_window_s=SMOOTH)
print("   Velocities shape:", df_v.shape)

print("5) Lade σ-Locks (Subset) ...")
locks = pd.read_csv("locks_storks_subset.csv")
print("   Locks-Zeilen:", len(locks))

def windows_for_pair(id_i, id_j):
    row = None
    for _, r in locks.iterrows():
        if (r["i"] == id_i and r["j"] == id_j) or (r["i"] == id_j and r["j"] == id_i):
            row = r
            break
    if row is None:
        print(f"   WARNUNG: keine Locks für Paar {id_i} - {id_j}")
        return []
    ep = row["episodes"]
    if isinstance(ep, str) and ep.strip():
        try:
            episodes = ast.literal_eval(ep)
        except Exception as e:
            print(f"   Fehler beim Parsen der episodes für {id_i}-{id_j}: {e}")
            return []
        print(f"   {len(episodes)} Lock-Fenster für Paar {id_i}-{id_j}")
        return episodes
    else:
        print(f"   Keine Episoden für Paar {id_i}-{id_j}")
        return []

print("6) TwistX für AB, BC, CA ...")
pairs = [("AB", ID_A, ID_B), ("BC", ID_B, ID_C), ("CA", ID_C, ID_A)]
twist_results = {}

for label, i, j in pairs:
    print(f"   -> {label}: {i} vs {j}")
    wins = windows_for_pair(i, j)
    res = compute_twistx(df_v, lock_windows=wins, id_i=i, id_j=j, eps=0.2, Nsurr=100)
    twist_results[label] = res
    out_name = f"twist_{label}.json"
    # numpy-Arrays in Listen umwandeln, damit JSON sie mag
    res_json = {
        "id_i": i,
        "id_j": j,
        "Stopo": float(res["Stopo"]),
        "ptwist": float(res["ptwist"]),
        "events_t": np.asarray(res["events_t"]).tolist(),
        "events_idx": np.asarray(res["events_idx"]).tolist(),
    }
    with open(out_name, "w") as f:
        json.dump(res_json, f, indent=2)
    print(f"      Stopo={res['Stopo']:.4f}, ptwist={res['ptwist']:.4f} -> {out_name}")

print("7) Triad-TwistX ...")
eAB = np.asarray(twist_results["AB"]["events_t"])
eBC = np.asarray(twist_results["BC"]["events_t"])
eCA = np.asarray(twist_results["CA"]["events_t"])
res_triad = triad_cycles(eAB, eBC, eCA, max_lag=2.0, Nsurr=300)

with open("triad_ABC.json", "w") as f:
    json.dump({
        "ID_A": ID_A,
        "ID_B": ID_B,
        "ID_C": ID_C,
        **{k: (float(v) if isinstance(v, (int,float,np.floating)) else v)
           for k,v in res_triad.items()}
    }, f, indent=2)

print("   Triad-Ergebnis:", res_triad)
print("FERTIG: twist_AB.json, twist_BC.json, twist_CA.json und triad_ABC.json geschrieben.")
