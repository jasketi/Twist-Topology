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
    # Alle Zahlen aus np.float64(...) holen
    nums_str = re.findall(r"np\.float64\(([-+]?\d*\.?\d+)\)", s)
    if not nums_str:
        return []
    nums = [float(x) for x in nums_str]
    if len(nums) < 2:
        return []
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
    required = ["i", "j", "episodes"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Spalte {col!r} fehlt in {csv_path}")
    episodes = []
    total_dur_per_edge = defaultdict(float)
    nodes = set()

    for _, row in df.iterrows():
        i = row["i"]
        j = row["j"]
        nodes.add(i)
        nodes.add(j)
        eps = parse_episodes_np64(row["episodes"])
        if not eps:
            continue
        for start, end in eps:
            episodes.append((i, j, start, end))
            total_dur_per_edge[(i, j)] += (end - start)

    if not episodes:
        raise RuntimeError("Keine Episoden aus episodes-Spalte geparst.")
    return episodes, sorted(nodes), total_dur_per_edge


def compute_density_timeseries(episodes, nodes, n_times=400):
    """
    Wählt dichte Kandidatenzeiten (Mittelpunkte der Episoden),
    berechnet Netzwerkdichte an jedem Zeitpunkt und gibt Zeitreihe + t* zurück.
    """
    # Kandidatenzeiten = Mittelpunkte aller Episoden
    mids = np.array([(s + e) / 2.0 for (_, _, s, e) in episodes], dtype=float)
    mids.sort()

    if len(mids) > n_times:
        # gleichmäßig ausdünnen
        idx = np.linspace(0, len(mids) - 1, n_times).astype(int)
        mids = mids[idx]

    densities = []
    active_edges_per_time = []

    for t in mids:
        # aktive Episoden an t
        active_edges = [(i, j) for (i, j, s, e) in episodes if s <= t <= e]
        active_edges_per_time.append(active_edges)

        # aktive Knoten
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


def build_snapshot_graph(episodes, total_dur_per_edge, t_star):
    """
    Baut Netzwerk-Graf zur Zeit t_star:
    - Knoten: alle Individuen, die zu dieser Zeit in irgendeiner Episode sind
    - Kanten: aktive Locks
    - Kanten-Gewicht: kumulative Dauer über alle Episoden (global)
    """
    G = nx.Graph()

    # aktive Episoden an t_star
    active_edges = []
    for (i, j, s, e) in episodes:
        if s <= t_star <= e:
            active_edges.append((i, j))

    if not active_edges:
        return G

    # Knoten & Kanten hinzufügen
    for i, j in active_edges:
        w = total_dur_per_edge.get((i, j), 0.0)
        G.add_edge(i, j, weight=w)

    return G


def plot_fig3b_storks(episodes, nodes, total_dur_per_edge, out_path="fig3b_storks_network_evolution.pdf"):
    mids, densities, active_edges_per_time = compute_density_timeseries(episodes, nodes)

    # Zeitpunkt maximaler Dichte
    idx_star = int(np.argmax(densities))
    t_star = mids[idx_star]

    # relative Zeiten in Stunden für die Achse
    t0 = mids[0]
    times_rel_hours = (mids - t0) / 3600.0
    t_star_rel = (t_star - t0) / 3600.0

    # Snapshot-Graf
    G = build_snapshot_graph(episodes, total_dur_per_edge, t_star)

    # Plot
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300})
    fig, (ax_time, ax_net) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # --- Panel A: Dichte-Zeitreihe ---
    ax_time.plot(times_rel_hours, densities, lw=1.0)
    ax_time.axvline(t_star_rel, color="gray", lw=0.7, linestyle="--")
    ax_time.set_xlabel("time (h, relative)")
    ax_time.set_ylabel("network density")
    ax_time.set_title("storks: density over time")

    # --- Panel B: Netzwerk-Snapshot ---
    if len(G.nodes) > 0:
        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Knotengrößen
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [80 + 320 * (degrees[n] / max_deg) for n in G.nodes]

        # Kantenbreiten nach globaler kumulativer Dauer
        weights = np.array([d.get("weight", 0.0) for (_, _, d) in G.edges(data=True)], dtype=float)
        if len(weights) > 0:
            max_w = weights.max()
            if max_w <= 0:
                widths = [0.8 for _ in weights]
            else:
                widths = list(0.5 + 3.0 * (weights / max_w))
        else:
            widths = []

        nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, ax=ax_net, width=widths)
        ax_net.set_title(f"storks: snapshot at t* (densest)")
        ax_net.set_axis_off()
    else:
        ax_net.text(0.5, 0.5, "no active locks at t*", ha="center", va="center", transform=ax_net.transAxes)
        ax_net.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Figure 3b geschrieben nach: {out_path}")


def main():
    print("Lade Storch-Episoden aus locks_storks_all.csv ...")
    episodes, nodes, total_dur_per_edge = load_stork_episodes("locks_storks_all.csv")
    print(f"{len(nodes)} Individuen, {len(episodes)} Episoden.")

    plot_fig3b_storks(episodes, nodes, total_dur_per_edge)


if __name__ == "__main__":
    main()
