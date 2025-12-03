#!/usr/bin/env python3
"""
Figure 1 (Vultures): σ-Lock detection and Δv-space for two example dyads.
Robuste Version: wählt einfach die zwei Dyaden mit dem größten Overlap.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Eingabedatei (HUJ Vultures)
INFILE = "Soaring flight in Eurasian griffon vultures (HUJ) (data from Harel and Nathan, 2018).csv"

# Ausgabe
OUTFILE = "figures/fig_sigma_examples_vultures.pdf"

# Parameter
SIGMA_THRESHOLD = 0.8   # Lock-Grenze
TIME_SCALE = 60.0       # Sekunden -> Minuten
TIME_LABEL = "time (min)"


def guess_col(df, logical_name, contains_list):
    """Findet eine Spalte anhand von Teilstrings."""
    cols = list(df.columns)
    lcols = [c.lower() for c in cols]
    for pat in contains_list:
        for c, lc in zip(cols, lcols):
            if pat in lc:
                print(f"[MAP] {logical_name} → '{c}' (match: '{pat}')")
                return c
    raise ValueError(f"Konnte keine Spalte für {logical_name} finden. Vorhandene Spalten: {cols}")


def load_vulture_data():
    print(f"[INFO] Lade: {INFILE}")
    df = pd.read_csv(INFILE)

    # Spalten erraten
    time_col = guess_col(df, "Zeit", ["time", "timestamp", "date"])
    id_col   = guess_col(df, "ID",   ["id", "individual"])
    lat_col  = guess_col(df, "Lat",  ["lat"])
    lon_col  = guess_col(df, "Lon",  ["lon", "long"])

    t_raw = pd.to_datetime(df[time_col])
    t_sec = (t_raw - t_raw.min()).dt.total_seconds().to_numpy()

    df2 = pd.DataFrame({
        "t_sec": t_sec,
        "id": df[id_col].astype(str),
        "lat": df[lat_col].astype(float),
        "lon": df[lon_col].astype(float),
    })

    print(f"[INFO] {df2['id'].nunique()} Individuen geladen, {len(df2)} Zeilen.")
    return df2


def latlon_to_xy(lat, lon):
    """Einfache XY-Projektion (Meter)."""
    R = 6371000.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    lat0 = np.nanmean(lat_r)
    x = R * (lon_r - np.nanmean(lon_r)) * np.cos(lat0)
    y = R * (lat_r - np.nanmean(lat_r))
    return x, y


def compute_velocity(df):
    """Berechnet vx, vy für jedes Individuum."""
    vel = {}
    for idv, d in df.groupby("id"):
        d = d.sort_values("t_sec")
        if len(d) < 10:
            continue

        x, y = latlon_to_xy(d["lat"].values, d["lon"].values)
        t = d["t_sec"].values

        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)

        dt[dt == 0] = np.nan
        vx = dx / dt
        vy = dy / dt

        t_mid = (t[:-1] + t[1:]) / 2.0

        dfv = pd.DataFrame({
            "t_sec": t_mid,
            "vx": vx,
            "vy": vy,
        }).dropna()

        if len(dfv) >= 20:
            vel[idv] = dfv

    print(f"[INFO] Geschwindigkeiten berechnet für {len(vel)} Individuen.")
    return vel


def find_top2_dyads(vel):
    """
    Berechnet Overlap für alle Paare, sortiert absteigend nach Overlap
    und gibt die Top 2 zurück (auch wenn Overlap relativ klein ist).
    """
    ids = list(vel.keys())
    overlaps = []

    print("[INFO] Berechne Overlap für alle Dyaden...")
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            d1 = vel[id1]
            d2 = vel[id2]
            joined = pd.merge(d1, d2, on="t_sec", how="inner", suffixes=("_1", "_2"))
            n = len(joined)
            if n > 0:
                overlaps.append((n, id1, id2, joined))

    if not overlaps:
        raise RuntimeError("Keine Dyaden mit gemeinsamem Zeitfenster gefunden.")

    overlaps.sort(reverse=True, key=lambda x: x[0])
    print("[INFO] Dyaden nach Overlap (Top 10):")
    for n, id1, id2, _ in overlaps[:10]:
        print(f"  {id1} × {id2}: {n} gemeinsame Punkte")

    top = overlaps[:2]
    return [(id1, id2, joined) for (n, id1, id2, joined) in top]


def compute_sigma(joined):
    """Berechnet σ_norm, is_lock, dvx, dvy."""
    vx1 = joined["vx_1"].values
    vy1 = joined["vy_1"].values
    vx2 = joined["vx_2"].values
    vy2 = joined["vy_2"].values

    dot = vx1 * vx2 + vy1 * vy2
    n1 = np.sqrt(vx1**2 + vy1**2)
    n2 = np.sqrt(vx2**2 + vy2**2)

    cos_theta = dot / (n1*n2 + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sigma_norm = 0.5 * (1.0 + cos_theta)

    df = joined.copy()
    df["sigma_norm"] = sigma_norm
    df["is_lock"] = df["sigma_norm"] >= SIGMA_THRESHOLD
    df["dvx"] = vx2 - vx1
    df["dvy"] = vy2 - vy1
    return df


def intervals(t, is_lock):
    """Lock-Intervalle bestimmen."""
    ints = []
    in_l = False
    start = None
    for i in range(len(t)):
        if is_lock[i] and not in_l:
            in_l = True
            start = t[i]
        elif not is_lock[i] and in_l:
            in_l = False
            ints.append((start, t[i]))
    if in_l:
        ints.append((start, t[-1]))
    return ints


def plot_sigma(ax, df, title):
    t = df["t_sec"].values
    t_plot = (t - t.min()) / TIME_SCALE
    s = df["sigma_norm"].values
    is_l = df["is_lock"].values

    ax.plot(t_plot, s, lw=1.0)
    for a, b in intervals(t_plot, is_l):
        ax.axvspan(a, b, color="lightgray", alpha=0.5)

    ax.axhline(SIGMA_THRESHOLD, ls="--", lw=0.8)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)
    ax.set_ylabel("σ_norm")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(TIME_LABEL)


def plot_dv(ax, df):
    ax.scatter(df["dvx"], df["dvy"], s=5, alpha=0.3)
    locked = df["is_lock"].values
    ax.scatter(df["dvx"][locked], df["dvy"][locked], s=10, alpha=0.8)
    ax.set_xlabel("Δv_x")
    ax.set_ylabel("Δv_y")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")


def main():
    df = load_vulture_data()
    vel = compute_velocity(df)
    if len(vel) < 2:
        raise RuntimeError("Zu wenige Individuen mit gültigen Geschwindigkeiten.")

    dyads = find_top2_dyads(vel)
    processed = [(id1, id2, compute_sigma(joined)) for (id1, id2, joined) in dyads]

    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5), constrained_layout=True)

    # Dyade 1
    (id1, id2, df1) = processed[0]
    plot_sigma(axes[0, 0], df1, f"{id1} × {id2}")
    plot_dv(axes[1, 0], df1)

    # Dyade 2 (falls vorhanden, sonst nochmal Dyade 1)
    if len(processed) > 1:
        (id1b, id2b, df2) = processed[1]
    else:
        (id1b, id2b, df2) = processed[0]

    plot_sigma(axes[0, 1], df2, f"{id1b} × {id2b}")
    plot_dv(axes[1, 1], df2)

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    fig.savefig(OUTFILE, dpi=300)
    print(f"[OK] Figure saved to {OUTFILE}")


if __name__ == "__main__":
    main()
