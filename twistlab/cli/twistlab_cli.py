
from __future__ import annotations
import argparse, json, sys, os
import pandas as pd
import numpy as np

from ..core.io import resample_positions, compute_velocities
from ..algorithms.sigma_lock import sigma_lock
from ..algorithms.twistx import compute_twistx
from ..algorithms.triad_twistx import triad_cycles
from ..algorithms.networks import build_lock_network, topo_centrality

def cmd_sigma(args):
    df = pd.read_csv(args.input)
    if {"lat","lon"}.issubset(df.columns) and {"x","y"}.isdisjoint(df.columns):
        # optional geo to metric
        from ..domains.movebank import latlon_to_xy
        x, y = latlon_to_xy(df["lat"].to_numpy(), df["lon"].to_numpy())
        df = df.copy()
        df["x"] = x; df["y"] = y
    df_r = resample_positions(df[["id","t","x","y"]], dt=args.dt)
    df_v = compute_velocities(df_r, dt=args.dt, smooth_window_s=args.smooth)
    out = sigma_lock(df_v, w_s=args.window, theta_lock=args.theta, min_dur_s=args.minlen,
                     Nsurr=args.nsurr, surrogate=args.surr, block=args.block)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} dyads)")

def cmd_twistx(args):
    df = pd.read_csv(args.input)
    df_r = resample_positions(df[["id","t","x","y"]], dt=args.dt)
    df_v = compute_velocities(df_r, dt=args.dt, smooth_window_s=args.smooth)
    # optional lock windows file
    lock_windows = None
    if args.locks:
        locks = pd.read_csv(args.locks)
        lock_windows = [(r["t_start"], r["t_end"]) for _, r in locks.iterrows()]
    res = compute_twistx(df_v, lock_windows=lock_windows, id_i=args.id_i, id_j=args.id_j,
                         eps=args.eps, Nsurr=args.nsurr)
    # write JSON
    with open(args.output, "w") as f:
        json.dump({"id_i": args.id_i, "id_j": args.id_j, **res}, f, indent=2)
    print(f"Wrote {args.output}")

def cmd_triad(args):
    # expects three JSON files from twistx with event times arrays
    import json
    with open(args.ab) as f: AB = json.load(f)
    with open(args.bc) as f: BC = json.load(f)
    with open(args.ca) as f: CA = json.load(f)
    res = triad_cycles(np.asarray(AB["events_t"]), np.asarray(BC["events_t"]), np.asarray(CA["events_t"]),
                       max_lag=args.maxlag, Nsurr=args.nsurr)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.output}")

def cmd_nets(args):
    locks = pd.read_csv(args.locks)
    twist = pd.read_csv(args.twist) if args.twist and os.path.exists(args.twist) else pd.DataFrame(columns=["i","j","ptwist"])
    # build simple centrality
    import networkx as nx
    G = build_lock_network(locks, p_thr=args.pthr)
    C = topo_centrality(locks, twist, p_thr=args.pthr)
    # export
    nx.write_weighted_edgelist(G, args.graph)
    C.to_csv(args.centrality, header=["C_topo"])
    print(f"Wrote graph: {args.graph} and centrality: {args.centrality}")

def main(argv=None):
    p = argparse.ArgumentParser(prog="twistlab", description="σ-Lock & TwistX toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sigma", help="Compute σ-Locks for all dyads")
    ps.add_argument("--input", "-i", required=True, help="CSV with columns id,t,x,y or id,t,lat,lon")
    ps.add_argument("--dt", type=float, required=True, help="target step (s)")
    ps.add_argument("--smooth", type=float, default=0.0, help="median smoothing window (s) for velocities")
    ps.add_argument("--window", type=float, default=5.0, help="σ median window (s)")
    ps.add_argument("--theta", type=float, default=0.2, help="σ threshold")
    ps.add_argument("--minlen", type=float, default=30.0, help="min lock duration (s)")
    ps.add_argument("--nsurr", type=int, default=100, help="surrogates")
    ps.add_argument("--surr", choices=["circular","block"], default="circular")
    ps.add_argument("--block", type=int, default=10, help="block size for block-shuffle")
    ps.add_argument("--output", "-o", required=True, help="locks.csv")
    ps.set_defaults(func=cmd_sigma)

    pt = sub.add_parser("twistx", help="Compute TwistX for one dyad")
    pt.add_argument("--input", "-i", required=True, help="CSV with columns id,t,x,y")
    pt.add_argument("--dt", type=float, required=True, help="target step (s)")
    pt.add_argument("--smooth", type=float, default=0.0, help="median smoothing window (s)")
    pt.add_argument("--id-i", dest="id_i", required=True, help="entity id i")
    pt.add_argument("--id-j", dest="id_j", required=True, help="entity id j")
    pt.add_argument("--eps", type=float, default=0.2, help="epsilon for recurrence")
    pt.add_argument("--nsurr", type=int, default=300, help="surrogates")
    pt.add_argument("--locks", help="optional locks.csv to restrict to windows")
    pt.add_argument("--output", "-o", required=True, help="twist.json")
    pt.set_defaults(func=cmd_twistx)

    pr = sub.add_parser("triad", help="Triad cycles & significance from three twist.json files")
    pr.add_argument("--ab", required=True, help="twist_AB.json")
    pr.add_argument("--bc", required=True, help="twist_BC.json")
    pr.add_argument("--ca", required=True, help="twist_CA.json")
    pr.add_argument("--maxlag", type=float, default=2.0, help="max lag between events (s)")
    pr.add_argument("--nsurr", type=int, default=300)
    pr.add_argument("--output", "-o", required=True, help="triad.json")
    pr.set_defaults(func=cmd_triad)

    pn = sub.add_parser("nets", help="Build lock network and C_topo")
    pn.add_argument("--locks", required=True, help="locks.csv (i,j,P_obs,p_lock,episodes)")
    pn.add_argument("--twist", help="optional twist.csv with ptwist per (i,j)")
    pn.add_argument("--pthr", type=float, default=0.05, help="p-value threshold")
    pn.add_argument("--graph", default="lock_graph.edgelist", help="output weighted edgelist")
    pn.add_argument("--centrality", default="centrality.csv", help="output centrality CSV")
    pn.set_defaults(func=cmd_nets)

    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
