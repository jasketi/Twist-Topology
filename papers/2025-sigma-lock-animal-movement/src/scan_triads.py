#!/usr/bin/env python3
import numpy as np
import pandas as pd

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.domains.movebank import latlon_to_xy
from twistlab.algorithms.triad_twistx import triad_cycles
from twistlab.algorithms.twistx import unwrap_phase, half_twist_indices

# --- Parameter, die du bei Bedarf anpassen kannst ---
DT = 1.0                 # Resampling-Schritt (s)
SMOOTH = 15.0            # Glättung der Geschwindigkeiten (s)
MAX_IDS = 10             # wie viele Störche für den Scan verwenden
MAXLAG = 2.0             # Triad-Toleranz (s)
NSURR_TRIAD = 500        # Surrogate für Triad-Signifikanz
MIN_EVENTS_PER_DYAD = 50 # Mindestanzahl Halb-Twists pro Dyade

print("1) Lese storks_positions.csv ...")
df = pd.read_csv("storks_positions.csv")
print("   Zeilen:", len(df))
print("   Spalten:", list(df.columns))

# 2) Projektion lat/lon -> x,y (falls nötig)
if {"lat","lon"}.issubset(df.columns) and {"x","y"}.isdisjoint(df.columns):
    print("2) lat/lon -> x,y ...")
    x, y = latlon_to_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
    df = df.copy()
    df["x"] = x
    df["y"] = y
    print("   fertig mit Projektion.")
elif not {"x","y"}.issubset(df.columns):
    raise ValueError("Erwarte Spalten x,y oder lat,lon in storks_positions.csv")

# 3) Resampling
print(f"3) Resampling auf dt = {DT} s ...")
df_r = resample_positions(df[["id","t","x","y"]], dt=DT)
print("   Shape nach Resampling:", df_r.shape)

# 4) Geschwindigkeiten
print(f"4) Geschwindigkeiten berechnen (smooth = {SMOOTH} s) ...")
df_v = compute_velocities(df_r, dt=DT, smooth_window_s=SMOOTH)
print("   Shape df_v:", df_v.shape)

# 5) Fokus-IDs bestimmen: Top-IDs nach lock_strength
print("5) IDs auswählen ...")
all_ids = sorted(df_v["id"].unique())
print("   Verfügbare IDs insgesamt:", len(all_ids))

try:
    ls = pd.read_csv("lock_strength_per_stork.csv")
    if "stork" not in ls.columns or "lock_strength" not in ls.columns:
        raise ValueError("lock_strength_per_stork.csv hat unerwartete Spalten.")
    ls = ls[ls["stork"].isin(all_ids)]
    ls = ls.sort_values("lock_strength", ascending=False)
    ids = ls["stork"].head(MAX_IDS).tolist()
    print("   Nutze Top-IDs nach lock_strength:")
    for s in ids:
        print("    -", s)
except Exception as e:
    print("   WARNUNG: Konnte lock_strength_per_stork.csv nicht verwenden:", e)
    print("   -> nehme die ersten IDs alphabetisch.")
    ids = all_ids[:MAX_IDS]
    for s in ids:
        print("    -", s)

# Hilfsfunktion: Halb-Twist-Events für eine Dyade
def dyad_events(df_vel: pd.DataFrame, id_i, id_j) -> np.ndarray:
    gi = df_vel[df_vel["id"] == id_i][["t","vx","vy"]].sort_values("t")
    gj = df_vel[df_vel["id"] == id_j][["t","vx","vy"]].sort_values("t")
    merged = gi.merge(gj, on="t", suffixes=("_i","_j"))
    if merged.empty:
        return np.array([], dtype=float)
    t = merged["t"].to_numpy()
    dvx = (merged["vx_j"] - merged["vx_i"]).to_numpy()
    dvy = (merged["vy_j"] - merged["vy_i"]).to_numpy()
    phi = np.arctan2(dvy, dvx)
    theta = unwrap_phase(phi)
    idx = half_twist_indices(theta)
    return t[idx] if idx.size > 0 else np.array([], dtype=float)

print("6) Halb-Twist-Events für alle Dyaden berechnen ...")
dyad_events_dict = {}
for a_idx in range(len(ids)):
    for b_idx in range(a_idx+1, len(ids)):
        A = ids[a_idx]
        B = ids[b_idx]
        eAB = dyad_events(df_v, A, B)
        dyad_events_dict[(A,B)] = eAB
        print(f"   Dyade {A} – {B}: {len(eAB)} Events")

print("7) Triaden scannen ...")
rows = []

for i_idx in range(len(ids)):
    for j_idx in range(i_idx+1, len(ids)):
        for k_idx in range(j_idx+1, len(ids)):
            A = ids[i_idx]
            B = ids[j_idx]
            C = ids[k_idx]

            def get_events(X, Y):
                key = (X, Y)
                if key in dyad_events_dict:
                    return dyad_events_dict[key]
                key_rev = (Y, X)
                if key_rev in dyad_events_dict:
                    return dyad_events_dict[key_rev]
                return np.array([], dtype=float)

            eAB = get_events(A, B)
            eBC = get_events(B, C)
            eCA = get_events(C, A)

            # nur Triaden mit ausreichend Events in jeder Dyade
            if min(len(eAB), len(eBC), len(eCA)) < MIN_EVENTS_PER_DYAD:
                continue

            res = triad_cycles(eAB, eBC, eCA, max_lag=MAXLAG, Nsurr=NSURR_TRIAD)
            rows.append({
                "A": A,
                "B": B,
                "C": C,
                "Striad": res["Striad"],
                "ptriad": res["ptriad"],
                "Ncycles": res["Ncycles"],
                "N_AB": len(eAB),
                "N_BC": len(eBC),
                "N_CA": len(eCA),
            })
            print(f"   Triade {A} – {B} – {C}: Striad={res['Striad']:.4g}, ptriad={res['ptriad']:.4g}")

if rows:
    triads_df = pd.DataFrame(rows)
    triads_df = triads_df.sort_values("ptriad")
    triads_df.to_csv("triads_scan.csv", index=False)
    print("8) FERTIG: triads_scan.csv mit", len(triads_df), "Triaden geschrieben.")
else:
    print("8) Keine Triaden mit ausreichend Events gefunden.")
