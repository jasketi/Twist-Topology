#!/usr/bin/env python3
"""
Figur 2: TwistX-State-Space-Plot für eine Griffon-Dyade

- Graue Linie: Trajektorie im Differential-Velocity-Space
- Blaue Punkte: Half-Twist-Events
- Oranger Stern: Start des Segments
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

# ------------------------------------------------------------
# 1) Daten einlesen
# ------------------------------------------------------------
pos_path = BASE / "griffons_positions.csv"
df = pd.read_csv(pos_path)

ID_COL = "id"
TIME_COL = "t"
LAT_COL = "lat"
LON_COL = "lon"

ids = sorted(df[ID_COL].unique())
if len(ids) < 2:
    raise SystemExit("Nicht genügend Individuen in griffons_positions.csv")

A_ID = ids[0]
B_ID = ids[1]
print(f"Verwende Dyade: {A_ID} - {B_ID}")

# ------------------------------------------------------------
# 2) Geschwindigkeiten pro Individuum
# ------------------------------------------------------------
def compute_velocity(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values(TIME_COL).copy()
    dt = sub[TIME_COL].diff()
    dlon = sub[LON_COL].diff()
    dlat = sub[LAT_COL].diff()
    sub["vx"] = dlon / dt
    sub["vy"] = dlat / dt
    sub = sub.dropna(subset=["vx", "vy"])
    return sub[[TIME_COL, "vx", "vy"]]

dfa = compute_velocity(df[df[ID_COL] == A_ID])
dfb = compute_velocity(df[df[ID_COL] == B_ID])

# ------------------------------------------------------------
# 3) Zeitlich mergen und Δv bilden
# ------------------------------------------------------------
merged = pd.merge(
    dfa,
    dfb,
    on=TIME_COL,
    suffixes=("_A", "_B")
)

if merged.empty:
    raise SystemExit("Kein gemeinsamer Zeitbereich für die Dyade gefunden.")

merged["dvx"] = merged["vx_B"] - merged["vx_A"]
merged["dvy"] = merged["vy_B"] - merged["vy_A"]

# ------------------------------------------------------------
# 4) TwistX-Winkel und Half-Twists
# ------------------------------------------------------------
dvx = merged["dvx"].values
dvy = merged["dvy"].values

phi = np.arctan2(dvy, dvx)
theta = np.unwrap(phi)
theta = theta - theta[0]

y = np.cos(theta / 2.0)
sign = np.sign(y)
sign[sign == 0] = 1
ht_indices = np.where(np.diff(sign) != 0)[0] + 1

merged["half_twist"] = False
merged.loc[ht_indices, "half_twist"] = True
print(f"Gefundene Half-Twist-Kandidaten: {len(ht_indices)}")

# ------------------------------------------------------------
# 5) Segment mit vielen Half-Twists wählen
# ------------------------------------------------------------
N = len(merged)
if len(ht_indices) >= 3:
    window = max(N // 10, 200)
    best_start = 0
    best_count = 0
    j = 0
    for i in range(len(ht_indices)):
        while j < len(ht_indices) and ht_indices[j] - ht_indices[i] <= window:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = ht_indices[i]
    start_idx = best_start
    end_idx = min(best_start + window, N - 1)
    segment = merged.iloc[start_idx:end_idx].copy()
    print(f"Verwende Segment [{start_idx}, {end_idx}] mit {best_count} Half-Twists.")
else:
    segment = merged.copy()
    print("Zu wenige Half-Twists – verwende gesamte Zeitreihe.")

segment_ht = segment[segment["half_twist"]].copy()

# ------------------------------------------------------------
# 6) Plot: breiter, mit großem weißen Rand rechts
# ------------------------------------------------------------
# Breiter Plot, damit rechts Platz für die Legende ist
fig, ax = plt.subplots(figsize=(6.5, 5))

x = segment["dvx"].values
y = segment["dvy"].values

# Trajektorie als graue Linie
h_traj, = ax.plot(
    x, y,
    linewidth=0.8,
    color="0.6"
)

# Half-Twists als blaue Punkte
h_ht = None
if not segment_ht.empty:
    h_ht = ax.scatter(
        segment_ht["dvx"].values,
        segment_ht["dvy"].values,
        s=30,
        color="tab:blue"
    )

# Startpunkt als oranger Stern
h_start = ax.scatter(
    x[0], y[0],
    s=40,
    marker="*",
    color="tab:orange",
    zorder=3
)

ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("Δv_x (relative longitudinal velocity)")
ax.set_ylabel("Δv_y (relative latitudinal velocity)")

# Viel Platz links für y-Label und rechts für Legende
fig.subplots_adjust(left=0.2, right=0.75, bottom=0.18, top=0.95)

# Legende im weißen Randbereich rechts
handles = [h_traj]
labels = ["trajectory"]
if h_ht is not None:
    handles.append(h_ht)
    labels.append("half-twist events")
handles.append(h_start)
labels.append("start")

ax.legend(
    handles,
    labels,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.
)

out_path = BASE / "fig_twistx_stateplots.pdf"
fig.savefig(out_path)
print(f"✔ fig_twistx_stateplots.pdf geschrieben")
