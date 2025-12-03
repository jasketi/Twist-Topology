#!/usr/bin/env python3
"""
Figure 1: Examples of σ-Lock detection in animal systems.

Dieses Skript:
- findet eine Positionsdatei (storks_positions.csv oder storks_gps.csv),
- errät Zeit-, ID-, Lat-, Lon-Spalten,
- berechnet pro Individuum Geschwindigkeiten (vx, vy),
- wählt zwei Dyaden mit ausreichend Überlappung,
- berechnet für jede Dyade:
    * sigma_norm(t) = (1 + cos(theta))/2, theta = Winkel zwischen v1 und v2
    * einfache Lock-Episoden (sigma_norm >= 0.8, zusammenhängend)
    * Differential velocity: Δv_x, Δv_y
- plottet 2x2:
    links Dyad 1, rechts Dyad 2
    oben: sigma_norm vs time mit schattierten Locks
    unten: Δv_x, Δv_y, Locks hervorgehoben
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ausgabedatei für die Figur
OUTFILE = "figures/fig_sigma_examples.pdf"

# Zeit-Skalierung: Sekunden -> Minuten
TIME_SCALE = 60.0
TIME_LABEL = "time since start (min)"

# Minimaler Überlapp (Anzahl Samples) für Dyaden
MIN_OVERLAP = 80

# Schwellwert für "Lock" (Richtungsalignment)
SIGMA_LOCK_THRESHOLD = 0.8


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------
def find_positions_file():
    """
    Versucht, eine Positions-Datei zu finden, die Lat/Lon + ID + Zeit enthält.
    Kandidaten: storks_positions.csv, storks_gps.csv
    """
    candidates = ["storks_positions.csv", "storks_gps.csv"]
    for fname in candidates:
        if not os.path.exists(fname):
            continue
        try:
            df = pd.read_csv(fname, nrows=200)
        except Exception:
            continue
        cols = [c.lower() for c in df.columns]
        has_lat = any("lat" in c for c in cols)
        has_lon = any("lon" in c or "long" in c for c in cols)
        has_id = any(c in cols for c in ["id", "track_id", "individual_id", "individual", "name"])
        has_time = any(c in cols for c in ["time", "timestamp", "datetime", "date_time", "t"])
        if has_lat and has_lon and has_id and has_time:
            print(f"[INFO] Verwende Positionsdatei: {fname}")
            return fname
    raise FileNotFoundError(
        "Konnte keine geeignete Positionsdatei finden. "
        "Erwartet z.B. storks_positions.csv oder storks_gps.csv mit Lat/Lon/ID/Zeit."
    )


def guess_col(df, logical, candidates_contains=None, candidates_exact=None):
    """
    Errät Spaltennamen:
    - candidates_exact: exakte Matches
    - candidates_contains: Teilstrings in Kleinbuchstaben
    """
    cols = list(df.columns)
    lcols = [c.lower() for c in cols]

    if candidates_exact:
        for ce in candidates_exact:
            for c, lc in zip(cols, lcols):
                if lc == ce.lower():
                    print(f"[MAP] {logical}: Spalte '{c}' (exakt '{ce}')")
                    return c

    if candidates_contains:
        for pattern in candidates_contains:
            for c, lc in zip(cols, lcols):
                if pattern.lower() in lc:
                    print(f"[MAP] {logical}: Spalte '{c}' (enthält '{pattern}')")
                    return c

    raise ValueError(f"Konnte keine Spalte für {logical} finden. Vorhandene Spalten: {cols}")


def load_positions():
    """Lädt Positionsdaten und gibt df mit Spalten: t_sec, id, lat, lon zurück."""
    fname = find_positions_file()
    df = pd.read_csv(fname)

    time_col = guess_col(
        df,
        "Zeit",
        candidates_exact=["timestamp", "time", "datetime", "date_time", "t"],
        candidates_contains=["time", "date", "utc"],
    )
    id_col = guess_col(
        df,
        "ID",
        candidates_exact=["id", "track_id", "individual_id", "individual", "name"],
        candidates_contains=["id", "name"],
    )
    lat_col = guess_col(
        df,
        "Latitude",
        candidates_exact=["lat", "latitude"],
        candidates_contains=["lat"],
    )
    lon_col = guess_col(
        df,
        "Longitude",
        candidates_exact=["lon", "long", "longitude"],
        candidates_contains=["lon"],
    )

    # Zeit in Sekunden relativ zum Start
    t_raw = df[time_col]
    if np.issubdtype(t_raw.dtype, np.number):
        t_sec = t_raw.to_numpy(dtype=float)
        t_sec = t_sec - np.nanmin(t_sec)
    else:
        t_dt = pd.to_datetime(t_raw)
        t_sec = (t_dt - t_dt.min()).dt.total_seconds().to_numpy(dtype=float)

    df_out = pd.DataFrame({
        "t_sec": t_sec,
        "id": df[id_col].astype(str),
        "lat": df[lat_col].astype(float),
        "lon": df[lon_col].astype(float),
    })

    print(f"[INFO] Positionsdaten geladen: {len(df_out)} Zeilen, {df_out['id'].nunique()} IDs")
    return df_out


def latlon_to_xy(lat, lon):
    """
    Sehr einfache Projektion: Lat/Lon -> x,y in Metern (equirectangular).
    """
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    R = 6371000.0  # Erdradius in m
    lat0 = np.nanmean(lat)
    x = R * (lon - lon.mean()) * np.cos(lat0)
    y = R * (lat - lat.mean())
    return x, y


def compute_velocity(df_pos):
    """
    df_pos: Spalten t_sec, id, lat, lon

    Gibt dict: id -> DataFrame mit t_sec, vx, vy
    """
    vel_by_id = {}
    for id_val, d in df_pos.groupby("id"):
        d = d.sort_values("t_sec")
        if len(d) < 5:
            continue
        x, y = latlon_to_xy(d["lat"].to_numpy(), d["lon"].to_numpy())
        t = d["t_sec"].to_numpy()
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)

        # kleine dt vermeiden
        dt[dt == 0] = np.nan
        vx = dx / dt
        vy = dy / dt

        # Zeitstempel für Geschwindigkeit: Mitte der Intervalle
        t_mid = (t[1:] + t[:-1]) / 2.0

        df_vel = pd.DataFrame({
            "t_sec": t_mid,
            "vx": vx,
            "vy": vy,
        }).dropna()

        if len(df_vel) < 20:
            continue

        vel_by_id[id_val] = df_vel

    print(f"[INFO] Geschwindigkeiten berechnet für {len(vel_by_id)} IDs")
    return vel_by_id


def find_example_dyads(vel_by_id, min_overlap=MIN_OVERLAP, max_pairs=2):
    """
    Sucht Dyaden mit ausreichend Überlappung in den Geschwindigkeitszeitreihen.
    Gibt Liste von (id1, id2, df_joined) zurück.
    """
    ids = list(vel_by_id.keys())
    pairs = []
    n = len(ids)
    print(f"[INFO] Suche Dyaden mit >= {min_overlap} gemeinsamen Zeitpunkten ...")
    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = ids[i], ids[j]
            d1 = vel_by_id[id1]
            d2 = vel_by_id[id2]

            # Merge auf t_sec (inner join)
            joined = pd.merge(d1, d2, on="t_sec", how="inner", suffixes=("_1", "_2"))
            if len(joined) >= min_overlap:
                print(f"  -> Dyade {id1} / {id2}: {len(joined)} gemeinsame Punkte")
                pairs.append((id1, id2, joined))
                if len(pairs) >= max_pairs:
                    return pairs
    if not pairs:
        raise RuntimeError(
            f"Konnte keine Dyade mit mindestens {min_overlap} gemeinsamen Punkten finden."
        )
    return pairs


def compute_sigma_and_dv(joined):
    """
    joined: DataFrame mit vx_1, vy_1, vx_2, vy_2

    Fügt Spalten hinzu:
    - sigma_norm
    - is_lock (simple Threshold-Regel)
    - dvx, dvy
    """
    vx1 = joined["vx_1"].to_numpy()
    vy1 = joined["vy_1"].to_numpy()
    vx2 = joined["vx_2"].to_numpy()
    vy2 = joined["vy_2"].to_numpy()

    v1 = np.stack([vx1, vy1], axis=1)
    v2 = np.stack([vx2, vy2], axis=1)

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    dot = (v1 * v2).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = dot / (n1 * n2)
    cos_theta[~np.isfinite(cos_theta)] = 0.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    sigma_norm = 0.5 * (1.0 + cos_theta)

    dvx = vx2 - vx1
    dvy = vy2 - vy1

    joined = joined.copy()
    joined["sigma_norm"] = sigma_norm
    joined["dvx"] = dvx
    joined["dvy"] = dvy

    joined["is_lock"] = joined["sigma_norm"] >= SIGMA_LOCK_THRESHOLD

    return joined


def lock_intervals(t, is_lock):
    """
    Aus einem bool-Array is_lock zusammenhängende Intervalle [t_start, t_end] bauen.
    """
    t = np.asarray(t)
    x = np.asarray(is_lock)

    intervals = []
    if len(t) == 0:
        return intervals

    in_lock = False
    start = None

    for i in range(len(t)):
        if x[i] and not in_lock:
            in_lock = True
            start = t[i]
        elif not x[i] and in_lock:
            in_lock = False
            end = t[i]
            intervals.append((start, end))

    if in_lock:
        intervals.append((start, t[-1]))

    return intervals


def plot_sigma_timeseries(ax, df, title):
    """Obere Panel: sigma_norm vs Zeit, Locks schattiert."""
    t_plot = (df["t_sec"].to_numpy() - df["t_sec"].min()) / TIME_SCALE
    s = df["sigma_norm"].to_numpy()
    is_lock = df["is_lock"].to_numpy()

    ax.plot(t_plot, s, lw=1.2)
    ax.set_ylabel(r"$\sigma_\mathrm{norm}$")
    ax.set_title(title, fontsize=9)

    intervals = lock_intervals(t_plot, is_lock)
    for t_start, t_end in intervals:
        ax.axvspan(t_start, t_end, color="lightgray", alpha=0.6)

    ax.axhline(SIGMA_LOCK_THRESHOLD, ls="--", lw=0.8)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3)


def plot_dv_state_space(ax, df):
    """Unteres Panel: Δv_x, Δv_y; Locks hervorgehoben."""
    dvx = df["dvx"].to_numpy()
    dvy = df["dvy"].to_numpy()
    is_lock = df["is_lock"].to_numpy()

    ax.scatter(dvx, dvy, s=5, alpha=0.3)
    ax.scatter(dvx[is_lock], dvy[is_lock], s=10, alpha=0.8)

    ax.set_xlabel(r"$\Delta v_x$")
    ax.set_ylabel(r"$\Delta v_y$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)


def main():
    # 1) Positionsdaten laden
    df_pos = load_positions()

    # 2) Geschwindigkeiten pro ID
    vel_by_id = compute_velocity(df_pos)
    if len(vel_by_id) < 2:
        raise RuntimeError("Zu wenige Individuen mit berechneten Geschwindigkeiten.")

    # 3) Dyaden mit ausreichender Überlappung finden
    pairs = find_example_dyads(vel_by_id, min_overlap=MIN_OVERLAP, max_pairs=2)

    # 4) sigma_norm + Δv für jede Dyade berechnen
    processed = []
    for (id1, id2, joined) in pairs:
        joined2 = compute_sigma_and_dv(joined)
        processed.append((id1, id2, joined2))

    # 5) Figur erzeugen (2x2)
    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.5),
        constrained_layout=True,
        sharex=False,
    )

    # Dyade 1
    (id1a, id2a, df1) = processed[0]
    ax1_top, ax1_bottom = axes[0, 0], axes[1, 0]
    plot_sigma_timeseries(ax1_top, df1, f"{id1a} / {id2a}")
    plot_dv_state_space(ax1_bottom, df1)
    ax1_bottom.set_xlabel(TIME_LABEL)

    # Dyade 2 (falls vorhanden), sonst Spiegelung
    if len(processed) > 1:
        (id1b, id2b, df2) = processed[1]
    else:
        (id1b, id2b, df2) = processed[0]

    ax2_top, ax2_bottom = axes[0, 1], axes[1, 1]
    plot_sigma_timeseries(ax2_top, df2, f"{id1b} / {id2b}")
    plot_dv_state_space(ax2_bottom, df2)
    ax2_bottom.set_xlabel(TIME_LABEL)

    # Ausgabe
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    fig.savefig(OUTFILE, dpi=300)
    print(f"\n[OK] Figure saved to: {OUTFILE}")


if __name__ == "__main__":
    main()
