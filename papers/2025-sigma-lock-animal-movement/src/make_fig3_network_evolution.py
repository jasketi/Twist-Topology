#!/usr/bin/env python3
"""
Figure 3: Lock-Network evolution across animal datasets.

Links:
    network density over time

Rechts:
    vultures : Snapshot-Netzwerk zum Dichte-Maximum
    storks   : Lock-Netzwerk in einem festen 1/10-Zeitfenster,
               das bei Minute 6 beginnt (relativ zum ersten Lock)

Knoten = Tiere, Kanten = Locks (Gewicht = kumulierte Lock-Dauer im Fenster).
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "figure.dpi": 300})

# ------------------------------------------------------------
# 1) Lock-Dateien finden
# ------------------------------------------------------------
csv_files = list(BASE.glob("*.csv"))

def find_file(patterns_any, must_contain="lock"):
    patterns_any = [p.lower() for p in patterns_any]
    for p in csv_files:
        name = p.name.lower()
        if must_contain and must_contain not in name:
            continue
        if any(word in name for word in patterns_any):
            return p
    return None

file_bats     = find_file(["bat"])
file_vultures = find_file(["griffon", "vulture"])
file_storks   = find_file(["stork"])

datasets = []
if file_bats is not None:
    datasets.append(("bats", file_bats))
if file_vultures is not None:
    datasets.append(("vultures", file_vultures))
if file_storks is not None:
    datasets.append(("storks", file_storks))

if not datasets:
    raise SystemExit("Keine passenden Lock-CSV-Dateien gefunden.")

print("Gefundene Lock-Dateien:")
for label, path in datasets:
    print(f"  {label}: {path.name}")

# ------------------------------------------------------------
# 2) Hilfsfunktionen
# ------------------------------------------------------------
def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)

    id_cols = [c for c in cols if re.search("id", c, re.IGNORECASE)]
    if len(id_cols) < 2:
        if len(cols) < 2:
            raise SystemExit("Zu wenige Spalten, um ID-Spalten zu finden.")
        id_cols = cols[:2]
    id1, id2 = id_cols[0], id_cols[1]

    start_candidates = [c for c in cols if re.search("start|t0|begin", c, re.IGNORECASE)]
    end_candidates   = [c for c in cols if re.search("end|t1|stop",   c, re.IGNORECASE)]
    if not start_candidates or not end_candidates:
        raise SystemExit("Start-/Endzeit-Spalten konnten nicht erkannt werden.")
    t_start, t_end = start_candidates[0], end_candidates[0]
    return id1, id2, t_start, t_end


def preprocess_df(df: pd.DataFrame, t_start_col: str, t_end_col: str):
    df = df.copy()
    df[t_start_col] = pd.to_datetime(df[t_start_col], errors="coerce")
    df[t_end_col]   = pd.to_datetime(df[t_end_col],   errors="coerce")
    df = df.dropna(subset=[t_start_col, t_end_col])
    df["weight"] = (df[t_end_col] - df[t_start_col]).dt.total_seconds()
    return df


def compute_density(df: pd.DataFrame,
                    id1_col: str,
                    id2_col: str,
                    t_start_col: str,
                    t_end_col: str,
                    n_bins: int = 40):
    all_ids = pd.unique(df[[id1_col, id2_col]].values.ravel())
    n_nodes = len(all_ids)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1

    t_min = df[t_start_col].min()
    t_max = df[t_end_col].max()
    if t_max <= t_min:
        raise ValueError("Zeitbereich degeneriert (t_max <= t_min).")

    bins = pd.interval_range(start=t_min, end=t_max, periods=n_bins)
    centers = []
    densities = []

    for interval in bins:
        sub = df[(df[t_start_col] < interval.right) & (df[t_end_col] > interval.left)]
        G = nx.Graph()
        for _, row in sub.iterrows():
            G.add_edge(row[id1_col], row[id2_col])
        m = G.number_of_edges()
        dens = m / max_edges if max_edges > 0 else 0.0
        densities.append(dens)
        centers.append(interval.left + (interval.right - interval.left) / 2)

    return np.array(centers), np.array(densities), t_min, t_max, all_ids


def build_snapshot_network(df: pd.DataFrame,
                           id1_col: str,
                           id2_col: str,
                           t_start_col: str,
                           t_end_col: str,
                           t_snap,
                           all_ids):
    sub = df[(df[t_start_col] <= t_snap) & (df[t_end_col] >= t_snap)]
    G = nx.Graph()
    G.add_nodes_from(all_ids)
    for _, row in sub.iterrows():
        u = row[id1_col]
        v = row[id2_col]
        w = row["weight"]
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def build_aggregated_network(df: pd.DataFrame,
                             id1_col: str,
                             id2_col: str,
                             all_ids):
    G = nx.Graph()
    G.add_nodes_from(all_ids)
    for _, row in df.iterrows():
        u = row[id1_col]
        v = row[id2_col]
        w = row["weight"]
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

# ------------------------------------------------------------
# 3) Figur zeichnen
# ------------------------------------------------------------
n_rows = len(datasets)
fig, axes = plt.subplots(n_rows, 2, figsize=(9.0, 3.0 * n_rows), squeeze=False)

for row_idx, (label, path) in enumerate(datasets):
    print(f"\nVerarbeite Dataset '{label}' aus {path.name}")
    df_raw = pd.read_csv(path)

    id1_col, id2_col, t_start_col, t_end_col = detect_columns(df_raw)
    df = preprocess_df(df_raw, t_start_col, t_end_col)
    centers, densities, t_min, t_max, all_ids = compute_density(
        df, id1_col, id2_col, t_start_col, t_end_col
    )

    # Dichte-Peak (für Vultures)
    idx_max = int(np.nanargmax(densities))
    t_peak = centers[idx_max]

    # ---------- Stork-Zeitfenster: 1/10 der Gesamtzeit, beginnend bei Minute 6 ----------
    window_start = None
    window_end = None
    if label == "storks":
        total_duration = t_max - t_min
        window_len = total_duration / 10.0

        # Start bei 6 Minuten nach t_min
        target_start = t_min + pd.to_timedelta(6, unit="m")
        window_start = target_start
        window_end   = window_start + window_len

        # Falls das über das Ende hinaus ragt, zurückschieben
        if window_end > t_max:
            window_end = t_max
            window_start = t_max - window_len

        if window_start < t_min:
            window_start = t_min

        print(f"  storks window: {window_start}  –  {window_end}")

    # ---------- links: Dichte ----------
    ax_dens = axes[row_idx, 0]
    delta_sec = np.array([(c - t_min).total_seconds() for c in centers])

    if label == "storks":
        x_vals = delta_sec / 60.0
        x_label = "time since start (min)"

        # 1/10-Fenster als graues Band
        ws = (window_start - t_min).total_seconds() / 60.0
        we = (window_end   - t_min).total_seconds() / 60.0
        ax_dens.axvspan(ws, we, color="0.85", alpha=0.4, zorder=0)
    else:
        x_vals = delta_sec / 3600.0
        x_label = "time since start (h)"

    ax_dens.plot(x_vals, densities, lw=0.9, zorder=1)

    if label == "vultures":
        x_peak = (t_peak - t_min).total_seconds() / 3600.0
        ax_dens.axvline(x_peak, linestyle="--", lw=0.7)

    ax_dens.set_xlabel(x_label)
    ax_dens.set_ylabel("network density")
    ax_dens.set_title(f"{label}: density over time")

    # ---------- rechts: Netzwerk ----------
    ax_net = axes[row_idx, 1]

    if label == "storks":
        # nur Locks im 1/10-Fenster ab Minute 6
        mask_win = (df[t_start_col] < window_end) & (df[t_end_col] > window_start)
        df_win = df[mask_win].copy()

        if df_win.empty:
            ax_net.text(0.5, 0.5, "no locks in window",
                        ha="center", va="center", transform=ax_net.transAxes)
            ax_net.set_axis_off()
        else:
            G_full = build_aggregated_network(df_win, id1_col, id2_col, all_ids)

            if G_full.number_of_edges() == 0:
                ax_net.text(0.5, 0.5, "no locks",
                            ha="center", va="center", transform=ax_net.transAxes)
                ax_net.set_axis_off()
            else:
                # größte zusammenhängende Komponente
                components = list(nx.connected_components(G_full))
                largest = max(components, key=len)
                Gplot = G_full.subgraph(largest).copy()

                # Layout etwas geordneter
                pos = nx.kamada_kawai_layout(Gplot)

                # nur stärkste ~3 % der Kanten behalten
                edges = list(Gplot.edges(data=True))
                weights = np.array([d.get("weight", 1.0) for (_, _, d) in edges], dtype=float)
                q = 0.97
                thr = np.quantile(weights, q) if len(weights) > 0 else 0.0
                keep_mask = weights >= thr
                edges_kept = [(u, v) for (u, v, _), k in zip(edges, keep_mask) if k]

                # Falls zu streng: fallback auf alle Kanten
                if not edges_kept:
                    edges_kept = [(u, v) for (u, v, _) in edges]
                    w_sel = weights
                else:
                    w_sel = weights[keep_mask]

                if len(edges_kept) > 0:
                    w_norm = (w_sel - w_sel.min()) / (w_sel.max() - w_sel.min() + 1e-9)
                    edge_widths = 0.4 + 1.8 * w_norm
                    nx.draw_networkx_edges(
                        Gplot, pos,
                        edgelist=edges_kept,
                        width=edge_widths,
                        edge_color="0.5",
                        ax=ax_net,
                    )

                degrees = dict(Gplot.degree())
                max_deg = max(degrees.values()) if degrees else 1
                node_sizes = [30 + 80 * (degrees[n] / max_deg) for n in Gplot.nodes()]

                nx.draw_networkx_nodes(
                    Gplot, pos,
                    node_size=node_sizes,
                    node_color="tab:blue",
                    ax=ax_net,
                )

                ax_net.set_title("storks: lock network (window from 6 min)")
                ax_net.set_axis_off()

    else:
        # vultures / bats: Snapshot beim Dichte-Maximum
        Gplot = build_snapshot_network(df, id1_col, id2_col,
                                       t_start_col, t_end_col,
                                       t_peak, all_ids)

        if Gplot.number_of_nodes() == 0:
            ax_net.text(0.5, 0.5, "no locks",
                        ha="center", va="center", transform=ax_net.transAxes)
            ax_net.set_axis_off()
        else:
            pos = nx.spring_layout(Gplot, seed=0)

            edges = list(Gplot.edges(data=True))
            if edges:
                weights = np.array([d.get("weight", 1.0) for (_, _, d) in edges], dtype=float)
                q = 0.4
                thr = np.quantile(weights, q) if len(weights) > 0 else 0.0
                keep_mask = weights >= thr
                edges_kept = [(u, v) for (u, v, _), k in zip(edges, keep_mask) if k]

                if edges_kept:
                    w_sel = weights[keep_mask]
                    w_norm = (w_sel - w_sel.min()) / (w_sel.max() - w_sel.min() + 1e-9)
                    edge_widths = 0.3 + 1.7 * w_norm
                    nx.draw_networkx_edges(
                        Gplot, pos,
                        edgelist=edges_kept,
                        width=edge_widths,
                        edge_color="0.5",
                        ax=ax_net,
                    )

            degrees = dict(Gplot.degree())
            max_deg = max(degrees.values()) if degrees else 1
            node_sizes = [25 + 80 * (degrees[n] / max_deg) for n in Gplot.nodes()]

            nx.draw_networkx_nodes(
                Gplot, pos,
                node_size=node_sizes,
                node_color="tab:blue",
                ax=ax_net,
            )

            ax_net.set_title(f"{label}: snapshot network")
            ax_net.set_axis_off()

fig.tight_layout()
out_path = BASE / "fig_network_evolution.pdf"
fig.savefig(out_path)
print(f"\n✔ fig_network_evolution.pdf geschrieben: {out_path}")
