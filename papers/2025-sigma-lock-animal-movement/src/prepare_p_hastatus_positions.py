#!/usr/bin/env python3
import pandas as pd
import numpy as np

INPUT_FILE = "Greater spear-nosed bat (Phyllostomus hastatus) in Bocas del Toro 2021-2022.csv"
OUTPUT_FILE = "p_hastatus_positions.csv"

print(f"1) Lese {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE)
print("   Zeilen gesamt:", len(df))
print("   Spalten:", list(df.columns))

# Nur GPS und sichtbare, nicht als Ausreißer markierte Punkte verwenden, falls Spalten existieren
if "sensor-type" in df.columns:
    df = df[df["sensor-type"] == "gps"]
if "visible" in df.columns:
    df = df[df["visible"] == True]
if "manually-marked-outlier" in df.columns:
    df = df[df["manually-marked-outlier"].isna()]

print("   Nach Filter:", len(df), "Zeilen")

# Zeitstempel in Sekunden seit Start
df["timestamp"] = pd.to_datetime(df["timestamp"])
t0 = df["timestamp"].min()
df["t"] = (df["timestamp"] - t0).dt.total_seconds().astype(float)

# ID, lat, lon auswählen
if "individual-local-identifier" in df.columns:
    df["id"] = df["individual-local-identifier"]
else:
    raise SystemExit("Spalte 'individual-local-identifier' nicht gefunden.")

df_out = df[["id", "t", "location-lat", "location-long"]].rename(
    columns={"location-lat": "lat", "location-long": "lon"}
).sort_values(["id", "t"])

print("   Output-Shape:", df_out.shape)
print("   Anzahl individueller IDs:", df_out['id'].nunique())
print("   IDs:", df_out['id'].unique())

print(f"2) Schreibe {OUTPUT_FILE} ...")
df_out.to_csv(OUTPUT_FILE, index=False)
print("FERTIG.")
