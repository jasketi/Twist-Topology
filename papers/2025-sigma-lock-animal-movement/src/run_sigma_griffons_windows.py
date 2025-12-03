#!/usr/bin/env python3
import pandas as pd

from twistlab.core.io import resample_positions, compute_velocities
from twistlab.domains.movebank import latlon_to_xy
from twistlab.algorithms.sigma_lock import sigma_lock

POS_FILE = "griffons_positions.csv"
OUTPUT_FILE = "locks_griffons_windows.csv"

# Drei vorgewählte Zeitfenster (UTC)
WINDOWS = [
    ("2013-08-31 07:08:00", "2013-08-31 10:10:00"),
    ("2013-09-01 06:32:00", "2013-09-01 11:34:00"),
    ("2013-09-02 06:25:00", "2013-09-02 11:26:00"),
]

DT = 1.0        # Zeitschritt in s
SMOOTH = 15.0   # Glättungsfenster für Geschw. in s
WINDOW_S = 60.0 # Sigma-Fenster in s
THETA = 0.2
MINLEN = 120.0
NSURR = 50
SURR_MODE = "circular"
BLOCK = 300.0

print(f"1) Lese {POS_FILE} ...")
df = pd.read_csv(POS_FILE)
print("   Shape:", df.shape)
print("   Spalten:", list(df.columns))

if "timestamp" not in df.columns:
    raise SystemExit("In griffons_positions.csv fehlt eine Spalte 'timestamp'.")

# WICHTIG: Strings -> echte Datumswerte
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"])

print("   IDs:", df["id"].nunique(), "->", df["id"].unique()[:10])
print("   Zeitspanne:", df["timestamp"].min(), "bis", df["timestamp"].max())

all_locks = []

for k, (start_str, end_str) in enumerate(WINDOWS, start=1):
    print(f"\n=== Fenster {k}: {start_str} bis {end_str} (UTC) ===")
    start = pd.to_datetime(start_str, utc=True)
    end = pd.to_datetime(end_str, utc=True)

    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df_win = df.loc[mask].copy()
    print("   Zeilen in diesem Fenster:", len(df_win))
    if df_win.empty:
        print("   -> Fenster leer, überspringe.")
        continue

    print("   IDs im Fenster:", df_win["id"].nunique(), "->", df_win["id"].unique())

    # Falls lat/lon noch nicht so heißen, anpassen
    if not {"lat","lon"}.issubset(df_win.columns):
        if {"location-lat","location-long"}.issubset(df_win.columns):
            df_win = df_win.rename(columns={
                "location-lat": "lat",
                "location-long": "lon",
            })
        else:
            raise SystemExit("Keine Spalten 'lat/lon' oder 'location-lat/-long' gefunden.")

    # 2) lat/lon -> x,y
    print("2) lat/lon -> x,y ...")
    x, y = latlon_to_xy(df_win["lat"].to_numpy(), df_win["lon"].to_numpy())
    df_win["x"] = x
    df_win["y"] = y
    print("   Projektion fertig.")

    # 3) t in Sekunden ab Fensteranfang
    t0 = df_win["timestamp"].min()
    df_win["t"] = (df_win["timestamp"] - t0).dt.total_seconds().astype(float)

    # 4) Resampling
    print(f"3) Resampling auf dt = {DT} s ...")
    df_r = resample_positions(df_win[["id","t","x","y"]], dt=DT)
    print("   Shape nach Resampling:", df_r.shape)
    print("   IDs nach Resampling:", df_r["id"].nunique())

    # 5) Geschwindigkeiten
    print(f"4) Geschwindigkeiten berechnen (smooth = {SMOOTH} s) ...")
    df_v = compute_velocities(df_r, dt=DT, smooth_window_s=SMOOTH)
    print("   Shape df_v:", df_v.shape)

    # 6) Sigma-Locks
    print("5) Sigma-Locks berechnen ...")
    locks = sigma_lock(
        df_v,
        w_s=WINDOW_S,
        theta_lock=THETA,
        min_dur_s=MINLEN,
        Nsurr=NSURR,
        surrogate=SURR_MODE,
        block=int(BLOCK),
        dt=DT,
    )
    print("   Anzahl Dyaden in diesem Fenster:", len(locks))

    if not locks.empty:
        locks["window_id"] = k
        locks["start_utc"] = start
        locks["end_utc"] = end
        all_locks.append(locks)

if all_locks:
    out = pd.concat(all_locks, ignore_index=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n6) FERTIG: {OUTPUT_FILE} mit", len(out), "Zeilen geschrieben.")
else:
    print("\n6) Keine Locks in den gewählten Fenstern gefunden – prüfe Zeitfenster/Parameter.")
