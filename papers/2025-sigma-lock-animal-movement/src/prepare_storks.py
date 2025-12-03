#!/usr/bin/env python3
import pandas as pd
import numpy as np

INPUT = "storks_gps.csv"       # ursprüngliche Movebank-GPS-Datei (umbenannt)
OUTPUT = "storks_positions.csv"

print(f"Lese {INPUT!r} ...")
df = pd.read_csv(INPUT)

print("Spalten im Input:")
print(list(df.columns))

# Kandidaten-Spalten definieren (typisches Movebank-Format)
time_cols = ["timestamp", "time", "event-time", "event_time"]
id_cols = ["individual-local-identifier", "individual_id", "animal-id", "id"]
lat_cols = ["location-lat", "location_lat", "latitude", "lat"]
lon_cols = ["location-long", "location_long", "longitude", "lon"]

def pick_column(kind, candidates):
    for c in candidates:
        if c in df.columns:
            print(f" -> benutze Spalte {c!r} als {kind}")
            return c
    raise SystemExit(
        f"FEHLER: Keine passende Spalte für {kind} gefunden.\n"
        f"Bitte sende mir diese Spaltenliste:\n{list(df.columns)}"
    )

time_col = pick_column("Zeit", time_cols)
id_col   = pick_column("Tier-ID", id_cols)
lat_col  = pick_column("Breite (lat)", lat_cols)
lon_col  = pick_column("Länge (lon)", lon_cols)

# Zeit in Sekunden seit Beginn
df["timestamp_parsed"] = pd.to_datetime(df[time_col])
t0 = df["timestamp_parsed"].min()
df["t"] = (df["timestamp_parsed"] - t0).dt.total_seconds()

# Minimales Ausgabeformat für twistlab
df["id"] = df[id_col].astype(str)
df["lat"] = df[lat_col].astype(float)
df["lon"] = df[lon_col].astype(float)

out = df[["id", "t", "lat", "lon"]].sort_values(["id", "t"]).reset_index(drop=True)

# Typisches Δt abschätzen (Median aller positiven Zeitabstände)
diffs = []
for _, g in out.groupby("id"):
    d = g["t"].diff().dropna()
    d = d[d > 0]
    diffs.extend(d.to_list())

if diffs:
    dt_med = float(np.median(diffs))
    print(f"Geschätztes typisches Δt = {dt_med:.1f} Sekunden")
    print(f"   = {dt_med/60:.2f} Minuten")
    # Parameter-Skalierung wie bei Sport (dt=0.2 -> 3s,5s,30s):
    smooth_s = 15 * dt_med
    window_s = 25 * dt_med
    minlen_s = 150 * dt_med
    maxlag_s = 2 * dt_med
    print("\nVORSCHLAG für σ-Lock-Parameter:")
    print(f"   --dt {dt_med:.1f}")
    print(f"   --smooth {smooth_s:.1f}")
    print(f"   --window {window_s:.1f}")
    print(f"   --minlen {minlen_s:.1f}")
    print("\nVORSCHLAG für Triad-Parameter:")
    print(f"   --maxlag {maxlag_s:.1f}")
else:
    print("WARNUNG: Konnte kein Δt schätzen (zu wenige Punkte pro Tier).")

out.to_csv(OUTPUT, index=False)
print(f"Fertig. Geschrieben: {OUTPUT!r} mit {out['id'].nunique()} Störchen und {len(out)} Punkten.")
