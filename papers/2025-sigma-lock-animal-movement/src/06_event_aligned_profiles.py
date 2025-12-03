#!/usr/bin/env python3
import pandas as pd
import numpy as np

WINDOW_STATES_FILE = "window_states.csv"
WINDOW_RADIUS = 5   # Anzahl Fenster vor/nach dem Event

print(f"1) Lese {WINDOW_STATES_FILE} ...")
ws = pd.read_csv(WINDOW_STATES_FILE)
print("   Zeilen:", len(ws))
print("   Spalten:", list(ws.columns))

if "FragState" not in ws.columns or "TriadState" not in ws.columns:
    raise SystemExit("Erwarte Spalten 'FragState' und 'TriadState' in window_states.csv")

frag = ws["FragState"].to_numpy()

# 2) Split- und Merge-Events finden
print("\n2) Suche Split- und Merge-Events ...")
events = []
for k in range(len(frag) - 1):
    if frag[k] == 0 and frag[k+1] == 1:
        events.append({"event_type": "split", "center_idx": k})
    if frag[k] == 1 and frag[k+1] == 0:
        events.append({"event_type": "merge", "center_idx": k})

print("   Gefundene Events:", len(events))
for e in events:
    print(f"      {e['event_type']} @ index {e['center_idx']}")

if not events:
    print("   Keine Events gefunden – nichts zu tun.")
    raise SystemExit()

# 3) Rohprofile um jedes Event (± WINDOW_RADIUS)
print("\n3) Event-Profile sammeln (±", WINDOW_RADIUS, "Fenster) ...")
rows = []
for event_id, ev in enumerate(events):
    center = int(ev["center_idx"])
    etype = ev["event_type"]
    for rel in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        idx = center + rel
        if idx < 0 or idx >= len(ws):
            continue
        w = ws.iloc[idx]
        rows.append({
            "event_id": event_id,
            "event_type": etype,
            "center_idx": center,
            "rel_idx": rel,
            "abs_idx": idx,
            "t_start": w.get("t_start", np.nan),
            "TriadState": w["TriadState"],
            "triad_count_sig": w.get("triad_count_sig", np.nan),
            "triad_S_sig_sum": w.get("triad_S_sig_sum", np.nan),
            "FragState": w["FragState"],
            "cluster_count": w.get("cluster_count", np.nan),
            "mean_distance": w.get("mean_distance", np.nan),
        })

raw = pd.DataFrame(rows)
raw = raw.sort_values(["event_type", "event_id", "rel_idx"])
raw.to_csv("event_profiles_raw.csv", index=False)
print("   event_profiles_raw.csv geschrieben mit", len(raw), "Zeilen.")

# 4) Zusammenfassung pro Event-Typ und Relativindex
print("\n4) Zusammenfassung pro (event_type, rel_idx) ...")
summary = (
    raw
    .groupby(["event_type", "rel_idx"])
    .agg(
        n_windows=("TriadState", "size"),
        mean_TriadState=("TriadState", "mean"),
        mean_triad_count_sig=("triad_count_sig", "mean"),
        mean_triad_S_sig_sum=("triad_S_sig_sum", "mean"),
        mean_FragState=("FragState", "mean"),
        mean_cluster_count=("cluster_count", "mean"),
        mean_mean_distance=("mean_distance", "mean"),
    )
    .reset_index()
    .sort_values(["event_type", "rel_idx"])
)

summary.to_csv("event_profiles_summary.csv", index=False)
print("   event_profiles_summary.csv geschrieben.")

print("\nFERTIG.")
