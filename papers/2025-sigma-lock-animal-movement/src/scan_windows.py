#!/usr/bin/env python3
import itertools
import os

import numpy as np
import pandas as pd
import networkx as nx

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.domains.movebank import latlon_to_xy
from twistlab.algorithms.twistx import unwrap_phase, half_twist_indices
from twistlab.algorithms.triad_twistx import triad_cycles

# ---------------- Parameter ----------------
DT = 1.0                 # Resampling-Schritt (s)
SMOOTH = 15.0            # Glättung der Geschwindigkeiten (s)
WINDOW_S = 1800.0        # Fensterlänge (z.B. 30 min)
STEP_S = 600.0           # Schritt zwischen Fenstern (z.B. 10 min)
MAX_IDS = 8              # wie viele Störche (Top-Triaden) verwenden
MAXLAG = 2.0             # Triad-Toleranz (s)
NSURR_TRIAD = 200        # Surrogate für Triad-Signifikanz
MIN_EVENTS_PER_DYAD = 20 # Mindestzahl Halb-Twists pro Dyade
DIST_THRESH = 5000.0     # Distanz-Schwelle für Cluster (Meter, ca. 5 km)

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

# 3) Fokus-IDs bestimmen (Top-Triaden aus triad_roles_per_stork.csv, falls vorhanden)
if os.path.exists("triad_roles_per_stork.csv"):
    print("3) Lese triad_roles_per_stork.csv, um Fokus-Störche zu wählen ...")
    roles = pd.read_csv("triad_roles_per_stork.csv")
    roles = roles.sort_values("mean_Striad", ascending=False)
    focus_ids = roles["stork"].head(MAX_IDS).tolist()
else:
    print("3) Keine triad_roles_per_stork.csv gefunden – nehme die ersten IDs aus den Daten.")
    focus_ids = sorted(df["id"].unique())[:MAX_IDS]

print("   Fokus-Störche:")
for s in focus_ids:
    print("    -", s)

df = df[df["id"].isin(focus_ids)].copy()

# 4) Globales Resampling & Geschwindigkeiten
print(f"4) Resampling auf dt = {DT} s ...")
df_r = resample_positions(df[["id","t","x","y"]], dt=DT)
print("   Shape df_r:", df_r.shape)

print(f"5) Geschwindigkeiten berechnen (smooth = {SMOOTH} s) ...")
df_v = compute_velocities(df_r, dt=DT, smooth_window_s=SMOOTH)
print("   Shape df_v:", df_v.shape)

ids = sorted(df_v["id"].unique())
print("   IDs nach Verarbeitung:", ids)

t_min = df_v["t"].min()
t_max = df_v["t"].max()
print(f"6) Zeitbereich: t_min={t_min:.1f}, t_max={t_max:.1f} (Dauer {(t_max - t_min)/3600:.2f} h)")

# Hilfsfunktion: Halb-Twist-Events für eine Dyade im gegebenen Fenster
def dyad_events_in_window(df_vel_w: pd.DataFrame, id_i, id_j) -> np.ndarray:
    gi = df_vel_w[df_vel_w["id"] == id_i][["t","vx","vy"]].sort_values("t")
    gj = df_vel_w[df_vel_w["id"] == id_j][["t","vx","vy"]].sort_values("t")
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

# Hilfsfunktion: Cluster auf Basis mittlerer Position pro Fenster
def swarm_metrics(df_pos_w: pd.DataFrame):
    if df_pos_w.empty:
        return np.nan, np.nan, 0
    # mittlere Position pro Storch im Fenster
    mean_pos = df_pos_w.groupby("id")[["x","y"]].mean().dropna()
    if len(mean_pos) < 2:
        return np.nan, np.nan, len(mean_pos)
    ids_local = mean_pos.index.tolist()
    X = mean_pos[["x","y"]].to_numpy()
    # paarweise Distanzen
    n = X.shape[0]
    dists = []
    G = nx.Graph()
    G.add_nodes_from(ids_local)
    for i in range(n):
        for j in range(i+1, n):
            dx = X[i,0] - X[j,0]
            dy = X[i,1] - X[j,1]
            d = np.hypot(dx, dy)
            dists.append(d)
            if d <= DIST_THRESH:
                G.add_edge(ids_local[i], ids_local[j])
    mean_dist = float(np.mean(dists)) if dists else np.nan
    components = list(nx.connected_components(G))
    cluster_count = len(components)
    return mean_dist, cluster_count, len(mean_pos)

rows = []
window_idx = 0
t0 = t_min

print("7) Sliding-Window-Analyse ...")
while t0 + WINDOW_S <= t_max:
    t1 = t0 + WINDOW_S
    window_idx += 1
    print(f"   Fenster {window_idx}: t = [{t0:.1f}, {t1:.1f})")

    # Teilmengen für dieses Fenster
    df_v_w = df_v[(df_v["t"] >= t0) & (df_v["t"] < t1)].copy()
    df_r_w = df_r[(df_r["t"] >= t0) & (df_r["t"] < t1)].copy()

    if df_v_w.empty or df_r_w.empty:
        print("      -> zu wenig Daten, überspringe Fenster.")
        t0 += STEP_S
        continue

    # Schwarm-Metriken
    mean_dist, cluster_count, n_ids_present = swarm_metrics(df_r_w)

    # Halb-Twist-Events für alle Dyaden in diesem Fenster
    dyad_events_dict = {}
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx+1, len(ids)):
            A = ids[a_idx]
            B = ids[b_idx]
            eAB = dyad_events_in_window(df_v_w, A, B)
            dyad_events_dict[(A,B)] = eAB

    # Triaden-Metriken
    triad_rows = []
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

                if min(len(eAB), len(eBC), len(eCA)) < MIN_EVENTS_PER_DYAD:
                    continue

                res = triad_cycles(eAB, eBC, eCA, max_lag=MAXLAG, Nsurr=NSURR_TRIAD)
                triad_rows.append({
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

    if triad_rows:
        tri_df = pd.DataFrame(triad_rows)
        triad_count_total = len(tri_df)
        triad_count_sig = int((tri_df["ptriad"] < 0.05).sum())
        triad_S_sum = float(tri_df["Striad"].sum())
        triad_S_mean = float(tri_df["Striad"].mean())
        triad_S_sig_sum = float(tri_df.loc[tri_df["ptriad"] < 0.05, "Striad"].sum())
    else:
        triad_count_total = 0
        triad_count_sig = 0
        triad_S_sum = np.nan
        triad_S_mean = np.nan
        triad_S_sig_sum = np.nan

    rows.append({
        "t_start": t0,
        "t_end": t1,
        "n_ids_present": n_ids_present,
        "mean_distance": mean_dist,
        "cluster_count": cluster_count,
        "triad_count_total": triad_count_total,
        "triad_count_sig": triad_count_sig,
        "triad_S_sum": triad_S_sum,
        "triad_S_mean": triad_S_mean,
        "triad_S_sig_sum": triad_S_sig_sum,
    })

    t0 += STEP_S

if rows:
    out_df = pd.DataFrame(rows)
    out_df.to_csv("window_metrics.csv", index=False)
    print("\n8) FERTIG: window_metrics.csv mit", len(out_df), "Fenstern geschrieben.")
else:
    print("\n8) Keine Fenster mit verwertbaren Daten gefunden.")
