
from __future__ import annotations
import pandas as pd
import networkx as nx

def build_lock_network(pairs_df: pd.DataFrame,
                       id_i: str = "i", id_j: str = "j",
                       weight_col: str = "P_obs",
                       p_col: str = "p_lock",
                       p_thr: float = 0.05) -> nx.Graph:
    """
    Build an undirected network from σ-Lock statistics.
    Edge kept if p_lock < p_thr; weight = weight_col.
    """
    G = nx.Graph()
    for _, row in pairs_df.iterrows():
        if row[p_col] < p_thr:
            i, j = row[id_i], row[id_j]
            w = float(row.get(weight_col, 1.0))
            G.add_edge(i, j, weight=w, p_lock=float(row[p_col]))
    return G

def topo_centrality(lock_df: pd.DataFrame, twist_df: pd.DataFrame,
                    id_i: str="i", id_j: str="j",
                    p_lock_col: str="p_lock", p_twist_col: str="ptwist",
                    p_thr: float=0.05) -> pd.Series:
    """
    C_topo_i = sum_j 1[p_lock<.05]*(1 - p_twist).
    Merge lock_df and twist_df by (i,j) (assume symmetric ordering already).
    """
    merged = lock_df.merge(twist_df[[id_i,id_j,p_twist_col]], on=[id_i,id_j], how="left")
    merged[p_twist_col] = merged[p_twist_col].fillna(1.0)
    merged["contrib"] = (merged[p_lock_col] < p_thr) * (1.0 - merged[p_twist_col])
    # accumulate to nodes
    vals = {}
    for _, r in merged.iterrows():
        i, j = r[id_i], r[id_j]
        c = float(r["contrib"])
        vals[i] = vals.get(i, 0.0) + c
        vals[j] = vals.get(j, 0.0) + c
    return pd.Series(vals).sort_values(ascending=False)
