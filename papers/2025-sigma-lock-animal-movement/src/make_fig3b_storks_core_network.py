import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


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
    mids = np.array([(s + e) / 2.0 for (_, _, s, e) in episodes], dtype=float)
    mids.sort()

    if len(mids) > n_times:
        idx = np.linspace(0, len(mids) - 1, n_times).astype(int)
        mids = mids[idx]

    densities = []
    for t in mids:
        active_edges = [(i, j) for (i, j, s, e) in episodes if s <= t <= e]
        active_nodes = set()
        for i, j in active_edges:
            active_nodes.add(i)
            active_nodes.add(j)
        n = len(active_nodes)
        m = len(active_edges)
        dens = 0.0 if n < 2 else 2.0 * m / (n * (n - 1))
        densities.append(dens)

    return mids, np.array(densities)


def choose_core_time(mids, densities, window_hours=2.0):
    if len(mids) == 0:
        raise RuntimeError("Keine Zeitpunkte für Dichteberechnung.")
    idx_star = int(np.argmax(densities))
    t_star = mids[idx_star]
    half = window_hours * 3600.0
    t_min = t_star - half
    t_max = t_star + half
    return t_star, t_min, t_max, idx_star


def build_window_edge_weights(episodes, t_star, half_window=1800.0):
    t0 = t_star - half_window
    t1 = t_star + half_window
    weights = defaultdict(float)
    nodes = set()

    for (i, j, s, e) in episodes:
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
    items = [(edge, dur) for edge, dur in weights.items() if dur > 0]
    if not items:
        return []
    durs = np.array([dur for (_, dur) in items], dtype=float)
    thr = np.quantile(durs, quantile)
    strong = [(edge, dur) for edge, dur in items if dur >= thr]
    strong.sort(key=lambda x: x[1], reverse=True)
    return strong[:max_edges]


def plot_core_figure(mids, densities, t_star, t_min, t_max, strong_edges, out_path):
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300})
    fig, (ax_time, ax_net) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # --- links: Dichte ---
    mask = (mids >= t_min) & (mids <= t_max)
    mids_win = mids[mask]
    dens_win = densities[mask]
    t_rel = (mids_win - t_star) / 3600.0

    ax_time.plot(t_rel, dens_win, lw=1.0)
    ax_time.axvline(0.0, linestyle="--", lw=0.7)
    ax_time.set_xlabel("time relative to t* (h)")
    ax_time.set_ylabel("network density")
    ax_time.set_title("storks: density around core episode")

    # --- rechts: Kernnetzwerk (spring layout) ---
    if not strong_edges:
        ax_net.text(0.5, 0.5, "no strong edges", ha="center", va="center",
                    transform=ax_net.transAxes)
        ax_net.set_axis_off()
    else:
        G = nx.Graph()
        for (i, j), dur in strong_edges:
            G.add_edge(i, j, weight=dur)

        pos = nx.spring_layout(G, seed=42)

        # Knotengrößen ∝ Grad
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1

        node_sizes = [30 + 90 * (degrees[n] / max_deg) for n in G.nodes]

        # Kantenbreite ∝ Dauer
        weights = np.array([d.get("weight", 0.0) for (_, _, d) in G.edges(data=True)], dtype=float)
        if len(weights) > 0:
            w_min = weights.min()
            w_max = weights.max()
            if w_max <= w_min:
                widths = [0.8 for _ in weights]
            else:
                widths = list(0.4 + 2.0 * (weights - w_min) / (w_max - w_min))
        else:
            widths = []

        nx.draw_networkx_edges(G, pos, ax=ax_net, width=widths, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=node_sizes)
        # Optional: Knoten-Beschriftungen (kann man im Zweifel wieder auskommentieren)
        nx.draw_networkx_labels(G, pos, ax=ax_net, font_size=6)

        ax_net.set_title("storks: core lock network (t*)")
        ax_net.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Figure 3b geschrieben nach: {out_path}")


def main():
    episodes, nodes = load_stork_episodes("locks_storks_all.csv")
    mids, densities = compute_density_timeseries(episodes)
    t_star, t_min, t_max, idx_star = choose_core_time(mids, densities, window_hours=2.0)
    print("t* (max density):", t_star, "density:", densities[idx_star])

    weights, nodes_win, (w0, w1) = build_window_edge_weights(
        episodes, t_star, half_window=1800.0
    )
    print("Fenster für Kernnetzwerk:", w0, "–", w1)
    strong_edges = select_top_edges(weights, quantile=0.7, max_edges=50)
    print("Anzahl starker Kanten:", len(strong_edges))

    out_name = "fig3b_storks_core_network.pdf"
    plot_core_figure(mids, densities, t_star, t_min, t_max, strong_edges, out_name)


if __name__ == "__main__":
    main()
