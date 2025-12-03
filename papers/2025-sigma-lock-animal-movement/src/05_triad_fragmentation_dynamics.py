#!/usr/bin/env python3
import pandas as pd
import numpy as np

# ---------------- Parameter ----------------
WINDOW_FILE = "window_metrics.csv"

# Triaden-Definition: wann nennen wir ein Fenster "Triaden-führend"?
# Hier: mindestens eine signifikante Triade
TRIAD_SIG_THRESHOLD = 0.05  # ptriad < 0.05 (informativ, aktuell nutzen wir triad_count_sig direkt)
MIN_TRIADS_SIG = 1          # mind. 1 signifikante Triade

# Fragmentierungs-Definition:
# FragState = 1, wenn cluster_count >= FRAG_CLUSTER_THR
# NEU: mildere Schwelle, schon ab 3 Clustern
FRAG_CLUSTER_THR = 3

print(f"1) Lese {WINDOW_FILE} ...")
df = pd.read_csv(WINDOW_FILE)
print("   Zeilen:", len(df))
print("   Spalten:", list(df.columns))

# 2) TriadState definieren
print("\n2) Triaden-Zustand pro Fenster bestimmen ...")
if "triad_count_sig" not in df.columns:
    raise SystemExit("Erwarte Spalte 'triad_count_sig' in window_metrics.csv")

df["TriadState"] = (df["triad_count_sig"] >= MIN_TRIADS_SIG).astype(int)

print("   Anzahl Fenster mit Triaden:", int(df["TriadState"].sum()))
print("   Anteil Fenster mit Triaden: "
      f"{df['TriadState'].mean():0.4f}")

# 3) Fragmentierungs-Zustand definieren
print("\n3) Fragmentierungs-Zustand pro Fenster bestimmen ...")
if "cluster_count" not in df.columns:
    raise SystemExit("Erwarte Spalte 'cluster_count' in window_metrics.csv")

df["FragState"] = (df["cluster_count"] >= FRAG_CLUSTER_THR).astype(int)

print("   FRAG_CLUSTER_THR =", FRAG_CLUSTER_THR)
print("   Anzahl fragmentierte Fenster (FragState=1):", int(df["FragState"].sum()))
print("   Anteil fragmentierte Fenster:", f"{df['FragState'].mean():0.4f}")

# 4) Fragmentierungs-Events: Übergang 0 -> 1
print("\n4) Fragmentierungs-Events (FragState 0 -> 1) suchen ...")
frag = df["FragState"].to_numpy()
t_start = df["t_start"].to_numpy()
events = []
for k in range(len(frag) - 1):
    if frag[k] == 0 and frag[k+1] == 1:
        events.append({
            "idx": k,
            "t_event": t_start[k+1],
            "Frag_before": int(frag[k]),
            "Frag_after": int(frag[k+1]),
            "cluster_before": int(df.loc[k, "cluster_count"]),
            "cluster_after": int(df.loc[k+1, "cluster_count"]),
            "mean_distance_before": df.loc[k, "mean_distance"],
            "mean_distance_after": df.loc[k+1, "mean_distance"],
        })

print("   Anzahl Fragmentierungs-Events:", len(events))
events_df = pd.DataFrame(events)
events_df.to_csv("fragmentation_events.csv", index=False)
print("   fragmentation_events.csv geschrieben.")

# 5) Vorwärts-Statistik: P(Fragmentierung_{k+1} | TriadState_k, FragState_k=0)
print("\n5) Vorwärts-Statistik berechnen ...")

triad = df["TriadState"].to_numpy()
frag = df["FragState"].to_numpy()

# Betrachte nur Fenster, in denen wir aktuell NICHT fragmentiert sind
idx_valid = np.where(frag[:-1] == 0)[0]
if len(idx_valid) == 0:
    print("   Keine Fenster mit FragState_k = 0 gefunden – nichts zu tun.")
else:
    triad_k = triad[idx_valid]
    frag_next = frag[idx_valid + 1]

    # Fälle mit Triaden vs. ohne Triaden
    mask_triad = triad_k == 1
    mask_no_triad = triad_k == 0

    def safe_frac(num, den):
        return float(num) / float(den) if den > 0 else np.nan

    # P(Fragmentation_{k+1}=1 | TriadState_k=1, FragState_k=0)
    n_triad = int(mask_triad.sum())
    n_triad_frag_next = int((mask_triad & (frag_next == 1)).sum())
    p_triad = safe_frac(n_triad_frag_next, n_triad)

    # P(Fragmentation_{k+1}=1 | TriadState_k=0, FragState_k=0)
    n_no_triad = int(mask_no_triad.sum())
    n_no_triad_frag_next = int((mask_no_triad & (frag_next == 1)).sum())
    p_no_triad = safe_frac(n_no_triad_frag_next, n_no_triad)

    print(f"   Fälle mit Triaden (TriadState_k=1, FragState_k=0): {n_triad}")
    print(f"      Davon mit FragState_{{k+1}}=1: {n_triad_frag_next}")
    print(f"      P(Fragmentierung im nächsten Fenster | Triaden aktiv) = {p_triad:0.4f}")
    print()
    print(f"   Fälle ohne Triaden (TriadState_k=0, FragState_k=0): {n_no_triad}")
    print(f"      Davon mit FragState_{{k+1}}=1: {n_no_triad_frag_next}")
    print(f"      P(Fragmentierung im nächsten Fenster | keine Triaden) = {p_no_triad:0.4f}")

# 6) Zustände mit abspeichern
out_states = df.copy()
out_states.to_csv("window_states.csv", index=False)
print("\n6) FERTIG: window_states.csv mit TriadState und FragState geschrieben.")
