#!/usr/bin/env python3
"""
Batch-Runs für Table 1 (Lorenz ramp: sensitivity of the sigma-lead Δρ).

- Lorenz-System mit ramped ρ(t) von 20 -> 35
- RK4-Integration
- Finite-Time Lyapunov (Benettin, zwei Trajektorien)
- Stroboscopic return (y,z) mit tau
- sigma-proxy δσ per closure-Fehler im Fenster
- Trigger-Zeiten:
    * t_sigma: erstes δσ < delta_sigma_thresh
    * t_ftle: erstes Lambda_W > 0
- Lead Δρ = ρ(t_ftle) - ρ(t_sigma)
- Ausgabe: Mediane, IQR, Erfolgsrate per Setting, als LaTeX-Zeilen
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def _load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML missing (needed for --config). Install: pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

import math
from dataclasses import dataclass, asdict

# ----------------- Parameter -----------------

RHO0 = 20.0
RHO1 = 35.0
SIGMA = 10.0
BETA = 8.0 / 3.0

DT = 0.005          # Integrationsschritt
K_REN = 10          # Renormalisierungsschritt für FTLE
FTLE_M = 50         # Anzahl Renorm-Schritte pro FTLE-Fenster
DELTA_SIGMA_THRESH = 0.25  # Trigger-Schwelle für δσ

W_STROBE = 300      # Anzahl Strobe-Punkte im Fenster
P_MAX = 8           # Max. Periode für Closure-Test

N_RUNS_DEFAULT = 50  # kannst du auf 100 hochsetzen, wenn es flott läuft


# ----------------- Numerik -----------------

def lorenz_rhs(state, t, rho0=RHO0, rho1=RHO1, T=300.0):
    """Lorenz RHS mit linear ramped ρ(t)."""
    x, y, z = state
    rho_t = rho0 + (rho1 - rho0) * (t / T)
    dx = SIGMA * (y - x)
    dy = x * (rho_t - z) - y
    dz = x * y - BETA * z
    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state, t, dt, rhs):
    k1 = rhs(state, t)
    k2 = rhs(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def compute_delta_sigma(window_xy, p_max=P_MAX):
    """
    δσ-Proxy im Sinn des Papers:
    - window_xy: np.ndarray, shape (W, 2), die letzten W Strobe-Punkte (y,z)
    - Rückgabe: δσ oder None
    """
    W = window_xy.shape[0]
    if W < W_STROBE:
        return None

    window = window_xy[-W_STROBE:]
    median_point = np.median(window, axis=0)
    dists_to_med = np.linalg.norm(window - median_point, axis=1)
    S = np.median(dists_to_med) + 1e-12

    half_idx = W_STROBE // 2
    best = np.inf
    for p in range(1, p_max + 1):
        if p >= W_STROBE:
            break
        start_k = max(half_idx, p)
        if start_k >= W_STROBE:
            continue
        diffs = window[start_k:] - window[start_k - p:W_STROBE - p]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) == 0:
            continue
        Ep = np.median(dists)
        val = Ep / S
        if val < best:
            best = val

    if best is np.inf:
        return None
    return best


@dataclass
class LeadResult:
    setting: str
    tau: float
    T: float
    delta_rho_median: float | None
    delta_rho_iqr: float | None
    success_fraction: float
    n_success: int
    n_runs: int


def simulate_single_run(T, tau, rng, delta_sigma_thresh=DELTA_SIGMA_THRESH):
    """
    Ein Lorenz-Ramp-Run mit:
    - stroboscopic σ-proxy
    - FTLE (Benettin)
    Liefert rho_sigma, rho_ftle (oder None, wenn Trigger nicht erreicht werden)
    """
    # Initialbedingungen leicht randomisiert, damit nicht alle Stichproben identisch sind
    x0 = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    state = x0.copy()
    t = 0.0

    # Zweite Trajektorie für FTLE
    eps0 = 1e-6
    state2 = state + eps0 * np.array([1.0, 0.0, 0.0])
    g_logs = []
    renorm_counter = 0

    rho_sigma = None
    rho_ftle = None

    # Stroboscopic sampling
    strobe_interval = tau
    next_strobe = t + strobe_interval
    strobe_points = []

    max_steps = int(T / DT)

    for step in range(max_steps):
        # RK4-Schritt für beide Trajektorien
        state = rk4_step(state, t, DT, lambda s, tt: lorenz_rhs(s, tt, T=T))
        state2 = rk4_step(state2, t, DT, lambda s, tt: lorenz_rhs(s, tt, T=T))
        t += DT
        renorm_counter += 1

        # FTLE: Benettin-Renormierung
        if renorm_counter >= K_REN:
            diff = state2 - state
            dist = np.linalg.norm(diff)
            if dist == 0.0:
                diff = eps0 * np.array([1.0, 0.0, 0.0])
                dist = np.linalg.norm(diff)
            g_logs.append(math.log(dist / eps0))
            # Renormalisieren
            state2 = state + eps0 * diff / dist
            renorm_counter = 0

            # Lokale FTLE-Schätzung über die letzten FTLE_M Renorm-Schritte
            m = len(g_logs)
            if m >= FTLE_M:
                window_logs = g_logs[-FTLE_M:]
                time_span = FTLE_M * K_REN * DT
                Lambda = sum(window_logs) / time_span
                if rho_ftle is None and Lambda > 0:
                    rho_t = RHO0 + (RHO1 - RHO0) * (t / T)
                    rho_ftle = rho_t

        # Stroboscopic return
        if t >= next_strobe - 1e-12:
            _, y, z = state
            strobe_points.append([y, z])
            next_strobe += strobe_interval

            points_arr = np.asarray(strobe_points, dtype=float)
            delta_sigma = compute_delta_sigma(points_arr)
            if rho_sigma is None and delta_sigma is not None and delta_sigma < delta_sigma_thresh:
                rho_t = RHO0 + (RHO1 - RHO0) * (t / T)
                rho_sigma = rho_t

    return rho_sigma, rho_ftle


def run_batch(setting_name, T, tau, n_runs=N_RUNS_DEFAULT, seed_base=0):
    rng = np.random.default_rng(seed_base)
    delta_rhos = []

    n_success = 0
    n_total = n_runs

    for k in range(n_runs):
        rho_sigma, rho_ftle = simulate_single_run(T=T, tau=tau, rng=rng)
        if rho_sigma is not None and rho_ftle is not None and rho_ftle > rho_sigma:
            delta_rhos.append(rho_ftle - rho_sigma)
            n_success += 1

    if n_success > 0:
        median = float(np.median(delta_rhos))
        q1 = float(np.quantile(delta_rhos, 0.25))
        q3 = float(np.quantile(delta_rhos, 0.75))
        iqr = q3 - q1
    else:
        median = None
        iqr = None

    success_fraction = n_success / n_total

    return LeadResult(
        setting=setting_name,
        tau=tau,
        T=T,
        delta_rho_median=median,
        delta_rho_iqr=iqr,
        success_fraction=success_fraction,
        n_success=n_success,
        n_runs=n_total,
    )


def format_latex_row(res: LeadResult) -> str:
    """Erzeugt eine LaTeX-Zeile im Table-1-Format (Variante C)."""
    if res.delta_rho_median is None:
        median_str = "n/a"
        iqr_str = "n/a"
    else:
        median_str = f"{res.delta_rho_median:.2f}"
        iqr_str = f"{res.delta_rho_iqr:.2f}"

    if res.success_fraction == 0.0:
        succ_str = "0.0"
    else:
        succ_str = f"{res.success_fraction:.2f}"

    return (
        f"{res.setting} & "
        f"{(f'{res.tau:.2f}' if res.tau is not None else '---')} & "
        f"{int(res.T)} & "
        f"{median_str} & "
        f"{iqr_str} & "
        f"{succ_str} & "
        f"$N={res.n_runs}$, "
        f"{res.n_success}~successes \\\\"
    )


def main():
    configs = [
        ("Strobe (default)", 0.20, 300.0),
        ("Strobe (finer)",   0.10, 300.0),
        ("Strobe (coarser)", 0.30, 300.0),
        ("Strobe, slower ramp", 0.20, 450.0),
        ("Strobe, faster ramp", 0.20, 200.0),
    ]

    results = []
    for name, tau, T in configs:
        print(f"Running setting: {name}, tau={tau}, T={T}, N={N_RUNS_DEFAULT} ...", flush=True)
        res = run_batch(name, T=T, tau=tau, n_runs=N_RUNS_DEFAULT)
        results.append(res)
        print("  ->", asdict(res))

    print("\nLaTeX rows for Table 1 (strobe settings):\n")
    for res in results:
        print(format_latex_row(res))

    print("\nDone.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (optional).")
    ap.add_argument("--outdir", default="papers/sigma-lock-ews/results")
    ap.add_argument("--n_runs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = _load_yaml_config(args.config) if args.config else {}

    # Allow config overrides for the existing global constants if present
    # (keeps your original implementation intact)
    global RHO0, RHO1, SIGMA, BETA, DT, K_REN, FTLE_M, DELTA_SIGMA_THRESH, W_STROBE, P_MAX, N_RUNS_DEFAULT
    if "lorenz" in cfg:
        lz = cfg["lorenz"]
        RHO0 = float(lz.get("rho0", RHO0))
        RHO1 = float(lz.get("rho1", RHO1))
        SIGMA = float(lz.get("sigma", SIGMA))
        BETA = float(lz.get("beta", BETA))
        DT = float(lz.get("dt", DT))
    if "ftle" in cfg:
        ft = cfg["ftle"]
        K_REN = int(ft.get("kren", K_REN))
        FTLE_M = int(ft.get("WFTLE", FTLE_M))
    if "return_proxy" in cfg:
        rp = cfg["return_proxy"]
        W_STROBE = int(rp.get("W", W_STROBE))
        P_MAX = int(rp.get("pmax", P_MAX))
        DELTA_SIGMA_THRESH = float(rp.get("delta_sigma_threshold", DELTA_SIGMA_THRESH))
    if "runs" in cfg:
        rr = cfg["runs"]
        N_RUNS_DEFAULT = int(rr.get("N", N_RUNS_DEFAULT))
        args.seed = int(rr.get("seed", args.seed))

    if args.n_runs is not None:
        N_RUNS_DEFAULT = int(args.n_runs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Choose settings: either from config or fall back to a small standard set.
    # Your original code likely expects run_batch(setting_name, T, tau, ...)
    settings = cfg.get("settings", None)
    if settings is None:
        # Default set; adjust to match your manuscript Table 1 once finalized.
        settings = [
            {"name": "baseline", "T": 300.0, "tau": 0.2},
            {"name": "tau=0.1", "T": 300.0, "tau": 0.1},
            {"name": "tau=0.3", "T": 300.0, "tau": 0.3},
            {"name": "T=200", "T": 200.0, "tau": 0.2},
        ]

    results = []
    for s in settings:
        res = run_batch(s.get("name","setting"), float(s.get("T",300.0)), float(s.get("tau",0.2)),
                        n_runs=N_RUNS_DEFAULT, seed_base=args.seed)
        results.append(res)

    # Write CSV
    csv_path = outdir / "table1_lorenz.csv"
    with csv_path.open("w") as f:
        f.write("setting,tau,T,delta_rho_median,delta_rho_iqr,success_fraction,n_success,n_runs\n")
        for r in results:
            f.write(
                f"{r.setting},{r.tau},{r.T},"
                f"{'' if r.delta_rho_median is None else r.delta_rho_median},"
                f"{'' if r.delta_rho_iqr is None else r.delta_rho_iqr},"
                f"{r.success_fraction},{r.n_success},{r.n_runs}\n"
            )

    # Write LaTeX rows
    tex_path = outdir / "table1_lorenz_rows.tex"
    tex_path.write_text("\n".join(format_latex_row(r) for r in results) + "\n")

    # Write run meta
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "config": args.config,
        "seed": args.seed,
        "n_runs": N_RUNS_DEFAULT,
        "settings": settings,
        "constants": {
            "RHO0": RHO0, "RHO1": RHO1, "SIGMA": SIGMA, "BETA": BETA,
            "DT": DT, "K_REN": K_REN, "FTLE_M": FTLE_M,
            "DELTA_SIGMA_THRESH": DELTA_SIGMA_THRESH, "W_STROBE": W_STROBE, "P_MAX": P_MAX
        },
    }
    (outdir / "run_meta_table1_lorenz.json").write_text(json.dumps(meta, indent=2))

    print("Wrote:", csv_path)
    print("Wrote:", tex_path)
    print("Wrote:", outdir / "run_meta_table1_lorenz.json")


if __name__ == "__main__":
    main()

