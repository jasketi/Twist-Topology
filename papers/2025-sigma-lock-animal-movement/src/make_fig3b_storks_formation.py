import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_episodes_np64(val):
    """
    Episodes-String wie:
    [(np.float64(2213984.0), np.float64(2214283.0)), ...]
    -> Liste von (start, end)-Floats
    """
    if pd.isna(val):
        return []
    s = str(val)
    # Alle Zahlen aus np.float64(...) holen
    nums_str = re.findall(r"np\.float64\(([-+]?\d*\.?\d+)\)", s)
    if not nums_str:
        return []
    nums = [float(x) for x in nums_str]
    # Sicherstellen, dass gerade Anzahl -> Paare
    if len(nums) % 2 == 1:
        nums = nums[:-1]
    pairs = []
    for a, b in zip(nums[0::2], nums[1::2]):
        if np.isnan(a) or np.isnan(b):
            continue
        start, end = (a, b) if a <= b else (b, a)
        if end > start:
            pairs.append((start, end))
    return pairs


def load_stork_episodes(csv_path="locks_storks_all.csv"):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    print("locks_storks_all columns:", cols)

    if "i" not in df.columns or "j" not in df.columns or "episodes" not in df.columns:
        raise RuntimeError("Erwarte Spalten 'i', 'j' und 'episodes' in locks_storks_all.csv")

    # optional: TwistX-p-Werte suchen
    twist_col = None
    for c in df.columns:
        if "ptwist" in c.lower():
            twist_col = c
            break

    episodes = []
    nodes = set()
    ptwist_map = {}

    for _, row in df.iterrows():
        i = row["i"]
        j = row["j"]
        nodes.add(i)
        nodes.add(j)

        eps = parse_episodes_np64(row["episodes"])
        if not eps:
            continue

        # TwistX-p-Wert (falls vorhanden)
        key = (i, j)
        if twist_col is not None:
            pt = row[twist_col]
            try:
                pt = float(pt)
            except Exception:
                pt = np.nan
            ptwist_map[key] = pt

        for start, end in eps:
            episodes.append((i, j, start, end))

    if not episodes:
        raise RuntimeError("Keine Episoden aus episodes-Spalte geparst.")
    print(f"{len(nodes)} Individuen, {len(episodes)} Episoden.")
    return episodes, sorted(nodes), ptwist_map


def compute_density_timeseries(episodes, n_times=400):
    """
    Kandidatenzeiten = Mittelpunkte aller Episoden.
    Für jede Zeit: aktive Kanten & Dichte berechnen.
    """
    mids = np.array([(s + e) / 2.0 for (_, _, s, e) in episodes], dtype=float)
    mids.sort()

    if len(mids) > n_times:
        idx = np.linspace(0, len(mids) - 1, n_times).astype(int)
        mids = mids[idx]

    densities = []
    active_edges_per_time = []

    for t in mids:
        active_edges = [(i, j) for (i, j, s, e) in episodes if s <= t <= e]
        active_edges_per_time.append(active_edges)

        active_nodes = set()
        for i, j in active_edges:
            active_nodes.add(i)
            active_nodes.add(j)

        n = len(active_nodes)
        m = len(active_edges)
        if n < 2:
            densities.append(0.0)
        else:
            densities.append(2.0 * m / (n * (n - 1)))

    densities = np.array(densities)
    return mids, densities, active_edges_per_time


def choose_two_times(mids, densities):
    """
    Wähle zwei Zeitpunkte:
    - t_low: sehr niedrige Dichte (unteres 10%-Quantil)
    - t_high: maximale Dichte
    """
    if len(mids) == 0:
        raise RuntimeError("Keine Zeitpunkte für Dichteberechnung.")

    idx_high = int(np.argmax(densities))
    t_high = mids[idx_high]

    # unteres Quantil für niedrige Dichte
    q10 = np.quantile(densities, 0.1)
    low_candidates = np.where(densities <= q10)[0]
    if len(low_candidates) == 0:
        idx_low = int(np.argmin(densities))
    else:
        idx_low = int(low_candidates[len(low_candidates) // 2])
    t_low = mids[idx_low]

    return t_low, t_high, idx_low, idx_high



def load_positions(pos_path, ids, t_target, dt_max=5.0):
    """
    Lese storks_positions.csv und hole die Position jedes ids zum t_target (±dt_max Sekunden).
    Erwartete Spalten: id, t, lat, lon.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(pos_path)
    print(f"{pos_path} columns:", list(df.columns))

    required = ["id", "t", "lat", "lon"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Erwarte Spalte {col!r} in {pos_path}")

    positions = []
    grouped = df.groupby("id")

    for id_ in ids:
        if id_ not in grouped.groups:
            continue
        sub = grouped.get_group(id_).copy()
        sub["dt"] = np.abs(sub["t"] - t_target)
        sub = sub.sort_values("dt")
        if sub["dt"].iloc[0] <= dt_max:
            row = sub.iloc[0]
            positions.append({
                "id": id_,
                "t": row["t"],
                "lat": row["lat"],
                "lon": row["lon"],
            })

    pos_df = pd.DataFrame(positions)
    return pos_df

def plot_formation(ax, pos_df, edges, ptwist_map=None, title=""):
    """
    Zeichne Formation mit Kanten:
    - normale Lock-Kanten in hellgrau
    - TwistX-signifikante Kanten (ptwist < 0.05) in dunkler Farbe
    Knoten-Größe ∝ Grad (Lock-Anzahl)
    """
    if pos_df.empty:
        ax.text(0.5, 0.5, "no positions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Mapping id -> (x, y)
    coords = {row["id"]: (row["lon"], row["lat"]) for _, row in pos_df.iterrows()}

    # Grad berechnen
    degree = defaultdict(int)
    for i, j in edges:
        if i in coords and j in coords:
            degree[i] += 1
            degree[j] += 1

    if degree:
        max_deg = max(degree.values())
    else:
        max_deg = 1

    # Plot Kanten
    if ptwist_map is None:
        ptwist_map = {}

    # Zuerst alle Lock-Kanten (hell)
    for i, j in edges:
        if i not in coords or j not in coords:
            continue
        x0, y0 = coords[i]
        x1, y1 = coords[j]
        ax.plot([x0, x1], [y0, y1], lw=0.4, alpha=0.4)

    # Dann TwistX-Kanten (falls es ptwist<0.05 gibt)
    twist_edges = [(i, j) for (i, j) in edges if ptwist_map.get((i, j), np.nan) < 0.05]
    for i, j in twist_edges:
        if i not in coords or j not in coords:
            continue
        x0, y0 = coords[i]
        x1, y1 = coords[j]
        ax.plot([x0, x1], [y0, y1], lw=1.2)

    # Knoten
    xs = []
    ys = []
    sizes = []
    for id_, (x, y) in coords.items():
        xs.append(x)
        ys.append(y)
        deg = degree.get(id_, 0)
        sizes.append(10 + 30 * (deg / max_deg if max_deg > 0 else 0.0))

    ax.scatter(xs, ys, s=sizes)

    ax.set_title(title)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal", adjustable="box")


def main():
    # --- Episoden & Dichte berechnen ---
    episodes, nodes, ptwist_map = load_stork_episodes("locks_storks_all.csv")
    mids, densities, active_edges_per_time = compute_density_timeseries(episodes)
    t_low, t_high, idx_low, idx_high = choose_two_times(mids, densities)
    print("t_low:", t_low, "  density_low:", densities[idx_low])
    print("t_high:", t_high, "  density_high:", densities[idx_high])

    # relative Zeiten in Stunden
    t0 = mids[0]
    t_rel_hours = (mids - t0) / 3600.0
    t_low_rel = (t_low - t0) / 3600.0
    t_high_rel = (t_high - t0) / 3600.0

    # --- Positionen laden ---
    # Aktive Knoten zu den beiden Zeitpunkten
    edges_low = active_edges_per_time[idx_low]
    edges_high = active_edges_per_time[idx_high]
    ids_low = sorted(set([i for (i, j) in edges_low] + [j for (i, j) in edges_low]))
    ids_high = sorted(set([i for (i, j) in edges_high] + [j for (i, j) in edges_high]))

    print(f"{len(ids_low)} aktive Individuen bei t_low, {len(ids_high)} bei t_high.")

    pos_low = load_positions("storks_positions.csv", ids_low, t_low, dt_max=300.0)
    pos_high = load_positions("storks_positions.csv", ids_high, t_high, dt_max=300.0)

    # --- Plot ---
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    ax_time, ax_low, ax_high = axes

    # Panel 1: Dichte-Zeitreihe
    ax_time.plot(t_rel_hours, densities, lw=1.0)
    ax_time.axvline(t_low_rel, linestyle="--", lw=0.7)
    ax_time.axvline(t_high_rel, linestyle="--", lw=0.7)
    ax_time.set_xlabel("time (h, relative)")
    ax_time.set_ylabel("network density")
    ax_time.set_title("storks: density over time")

    # Panel 2: Formation bei niedriger Dichte
    title_low = "storks: formation (low density)"
    plot_formation(ax_low, pos_low, edges_low, ptwist_map, title=title_low)

    # Panel 3: Formation bei hoher Dichte
    title_high = "storks: formation (high density)"
    plot_formation(ax_high, pos_high, edges_high, ptwist_map, title=title_high)

    fig.tight_layout()
    out_name = "fig3b_storks_formation.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    print(f"✅ Figure 3b geschrieben nach: {out_name}")


if __name__ == "__main__":
    main()
