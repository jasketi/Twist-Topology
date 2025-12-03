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
    nums_str = re.findall(r"np\.float64\(([-+]?\d*\.?\d+)\)", s)
    if not nums_str:
        return []
    nums = [float(x) for x in nums_str]
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
    print("locks_storks_all columns:", list(df.columns))

    for col in ["i", "j", "episodes"]:
        if col not in df.columns:
            raise RuntimeError(f"Erwarte Spalte {col!r} in {csv_path}")

    episodes = []
    nodes = set()

    for _, row in df.iterrows():
        i = row["i"]
        j = row["j"]
        nodes.add(i)
        nodes.add(j)
        eps = parse_episodes_np64(row["episodes"])
        for start, end in eps:
            episodes.append((i, j, start, end))

    if not episodes:
        raise RuntimeError("Keine Episoden aus episodes-Spalte geparst.")
    print(f"{len(nodes)} Individuen, {len(episodes)} Episoden.")
    return episodes, sorted(nodes)


def compute_density_timeseries(episodes, n_times=400):
    """
    Kandidatenzeiten = Mittelpunkte aller Episoden.
    Für jede Zeit: aktive Kanten & Netzwerkdichte.
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


def choose_high_density_window(mids, densities, window_hours=2.0):
    """
    Wähle t* (max. Dichte) und ein Zoom-Fenster [t_min, t_max] um t*.
    """
    if len(mids) == 0:
        raise RuntimeError("Keine Zeitpunkte für Dichteberechnung.")
    idx_star = int(np.argmax(densities))
    t_star = mids[idx_star]
    half = window_hours * 3600.0
    t_min = t_star - half
    t_max = t_star + half
    return t_star, t_min, t_max, idx_star


def build_window_edge_weights(episodes, t_star, half_window=1800.0):
    """
    Akkumuliere Lock-Dauer pro Dyade im Fenster [t*-half_window, t*+half_window].
    """
    t0 = t_star - half_window
    t1 = t_star + half_window
    weights = defaultdict(float)
    nodes = set()

    for (i, j, s, e) in episodes:
        # Überlappung mit dem Fenster
        overlap_start = max(s, t0)
        overlap_end = min(e, t1)
        if overlap_end <= overlap_start:
            continue
        dur = overlap_end - overlap_start
        weights[(i, j)] += dur
        nodes.add(i)
        nodes.add(j)

    return weights, nodes, (t0, t1)


def select_top_edges(weights, quantile=0.7, max_edges=50):
    """
    Wähle die stärksten Kanten nach Dauer:
    - nur Kanten mit Dauer > 0
    - Schwellwert = Quantil
    - max_edges begrenzt die Anzahl
    """
    items = [(edge, dur) for edge, dur in weights.items() if dur > 0]
    if not items:
        return []

    durs = np.array([dur for (_, dur) in items], dtype=float)
    thr = np.quantile(durs, quantile)
    strong = [(edge, dur) for edge, dur in items if dur >= thr]
    # nach Dauer sortieren, absteigend
    strong.sort(key=lambda x: x[1], reverse=True)
    return strong[:max_edges]


def load_positions(pos_path, ids, t_target, dt_max=300.0):
    """
    Lese storks_positions.csv und hole die Position jedes ids zum t_target (±dt_max Sekunden).
    Erwartete Spalten: id, t, lat, lon.
    """
    df = pd.read_csv(pos_path)
    print(f"{pos_path} columns:", list(df.columns))

    for col in ["id", "t", "lat", "lon"]:
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


def plot_core_figure(mids, densities, t_star, t_min, t_max,
                     strong_edges, positions_df, out_path):
    """
    Links: gezoomte Dichtekurve um t*.
    Rechts: Formation + Top-Kanten.
    """
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300})
    fig, (ax_time, ax_net) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # --- Panel links: Dichte ---
    mask = (mids >= t_min) & (mids <= t_max)
    mids_win = mids[mask]
    dens_win = densities[mask]

    # Zeiten relativ zu t_star (in Stunden)
    t_rel = (mids_win - t_star) / 3600.0

    ax_time.plot(t_rel, dens_win, lw=1.0)
    ax_time.axvline(0.0, linestyle="--", lw=0.7)
    ax_time.set_xlabel("time relative to t* (h)")
    ax_time.set_ylabel("network density")
    ax_time.set_title("storks: density around core episode")

    # --- Panel rechts: Formation + Netzwerk ---
    if positions_df.empty or not strong_edges:
        ax_net.text(0.5, 0.5, "no positions / edges", ha="center", va="center",
                    transform=ax_net.transAxes)
        ax_net.set_axis_off()
    else:
        coords = {row["id"]: (row["lon"], row["lat"]) for _, row in positions_df.iterrows()}

        # Grad nach starker Kantenliste
        degree = defaultdict(int)
        for (i, j), dur in strong_edges:
            if i in coords and j in coords:
                degree[i] += 1
                degree[j] += 1

        max_deg = max(degree.values()) if degree else 1

        # Kanten: Linienbreite ∝ Dauer
        durs = np.array([dur for (_, dur) in strong_edges], dtype=float)
        d_min = durs.min()
        d_max = durs.max() if durs.max() > d_min else d_min + 1.0

        for (i, j), dur in strong_edges:
            if i not in coords or j not in coords:
                continue
            x0, y0 = coords[i]
            x1, y1 = coords[j]
            # normierte Dauer
            alpha = (dur - d_min) / (d_max - d_min)
            lw = 0.4 + 2.0 * alpha
            ax_net.plot([x0, x1], [y0, y1], lw=lw, alpha=0.7)

        # Knoten
        xs, ys, sizes = [], [], []
        for id_, (x, y) in coords.items():
            xs.append(x)
            ys.append(y)
            deg = degree.get(id_, 0)
            sizes.append(10 + 35 * (deg / max_deg if max_deg > 0 else 0.0))

        ax_net.scatter(xs, ys, s=sizes)
        ax_net.set_title("storks: core lock network (t*)")
        ax_net.set_xlabel("lon")
        ax_net.set_ylabel("lat")
        ax_net.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Figure 3b geschrieben nach: {out_path}")


def main():
    # Episoden laden
    episodes, nodes = load_stork_episodes("locks_storks_all.csv")

    # Dichte über Zeit
    mids, densities, _ = compute_density_timeseries(episodes)

    # t* und Zoom-Fenster wählen
    t_star, t_min, t_max, idx_star = choose_high_density_window(
        mids, densities, window_hours=2.0
    )
    print("t* (max density):", t_star, "density:", densities[idx_star])
    print("Zoom window [t_min, t_max]:", t_min, t_max)

    # Kanten-Dauern in ±30min um t*
    weights, nodes_win, (w0, w1) = build_window_edge_weights(
        episodes, t_star, half_window=1800.0
    )
    print("Fenster für Kernnetzwerk:", w0, "–", w1)
    print("Anzahl Nodes im Fenster:", len(nodes_win))

    strong_edges = select_top_edges(weights, quantile=0.7, max_edges=50)
    print("Anzahl starker Kanten:", len(strong_edges))

    ids_core = sorted(set([i for (i, j), _ in strong_edges] +
                          [j for (i, j), _ in strong_edges]))
    print("IDs im Kernnetzwerk:", ids_core)

    # Positionen zum Zeitpunkt t* laden
    positions_df = load_positions("storks_positions.csv", ids_core, t_star, dt_max=600.0)

    # Figur plotten
    out_name = "fig3b_storks_core.pdf"
    plot_core_figure(mids, densities, t_star, t_min, t_max,
                     strong_edges, positions_df, out_name)


if __name__ == "__main__":
    main()
