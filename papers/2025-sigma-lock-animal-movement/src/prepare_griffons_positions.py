#!/usr/bin/env python3
import pandas as pd

INPUT_FILE = "Soaring flight in Eurasian griffon vultures (HUJ) (data from Harel and Nathan, 2018).csv"
OUTPUT_FILE = "griffons_positions.csv"

print(f"1) Lese {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
print("   Zeilen gesamt:", len(df))
print("   Spalten:", list(df.columns))

# Nur GPS, sichtbare, nicht als Ausreißer markierte Punkte (falls Spalten existieren)
if "sensor-type" in df.columns:
    df = df[df["sensor-type"] == "gps"]
if "visible" in df.columns:
    df = df[df["visible"] == True]
if "manually-marked-outlier" in df.columns:
    df = df[df["manually-marked-outlier"].isna()]

print("   Nach Filter:", len(df), "Zeilen")

# Zeitstempel in UTC-Datetime
if "timestamp" not in df.columns:
    raise SystemExit("Spalte 'timestamp' nicht gefunden – ist das eine Movebank-CSV?")

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sekundenskala t ab erstem Zeitpunkt
t0 = df["timestamp"].min()
df["t"] = (df["timestamp"] - t0).dt.total_seconds().astype(float)

# Individuen-ID
if "individual-local-identifier" in df.columns:
    df["id"] = df["individual-local-identifier"]
else:
    raise SystemExit("Spalte 'individual-local-identifier' nicht gefunden.")

# Lat/Lon
if not {"location-lat","location-long"}.issubset(df.columns):
    raise SystemExit("Erwarte 'location-lat' und 'location-long' in der Datei.")

df_out = (
    df[["id","t","timestamp","location-lat","location-long"]]
    .rename(columns={"location-lat":"lat","location-long":"lon"})
    .sort_values(["id","t"])
)

print("   Output-Shape:", df_out.shape)
print("   Anzahl individueller IDs:", df_out["id"].nunique())
print("   IDs (erste 10):", df_out["id"].unique()[:10])

print(f"2) Schreibe {OUTPUT_FILE} ...")
df_out.to_csv(OUTPUT_FILE, index=False)
print("FERTIG.")
