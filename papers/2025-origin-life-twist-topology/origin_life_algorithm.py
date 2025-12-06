"""
Origin-of-life algorithm for the paper
"Twist-Topological Constraints on the Origin of Life:
 Why Life Cannot Be Causally Constructed but Only Probabilistically Enabled".

This module implements a minimal, self-contained example:

- a 3-species autocatalytic reaction network (A, B, C),
- simple time-dependent "twist" in the reaction rates,
- a σ-locking index L(t) that is high when the dynamics is slow and
  the three species are in balance, low when the system is far from
  any quasi-locked regime.

It produces a pandas.DataFrame with columns:
time, A, B, C, L

and is intended as a transparent, reproducible reference implementation
for the conceptual framework in the paper. It does NOT depend on the
shared twistlab package.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class EnvironmentParams:
    """Parameters controlling the environment and twist driving."""
    k1: float = 1.0          # base rate A + B -> 2B
    k2: float = 1.0          # base rate B + C -> 2C
    k3: float = 1.0          # base rate C + A -> 2A
    decay: float = 0.1       # linear decay of each species
    inflow: float = 0.5      # constant inflow / background production
    twist_amplitude: float = 0.3  # amplitude of twist modulation
    twist_period: float = 50.0    # period of twist modulation (time units)


def twist_factor(t: float, env: EnvironmentParams) -> float:
    """Simple periodic twist factor between (1 - amp) and (1 + amp)."""
    return 1.0 + env.twist_amplitude * np.sin(2.0 * np.pi * t / env.twist_period)


def reaction_rhs(state: np.ndarray, t: float, env: EnvironmentParams) -> np.ndarray:
    """
    Right-hand side of the ODE system for (A, B, C).

    dA/dt = k3_eff * C*B - decay*A + inflow
    dB/dt = k1_eff * A*C - decay*B + inflow
    dC/dt = k2_eff * A*B - decay*C + inflow

    where the effective rates k*_eff are modulated by the twist factor.
    """
    A, B, C = state
    tf = twist_factor(t, env)
    k1_eff = env.k1 * tf
    k2_eff = env.k2 * tf
    k3_eff = env.k3 * tf

    dA = k3_eff * C * B - env.decay * A + env.inflow
    dB = k1_eff * A * C - env.decay * B + env.inflow
    dC = k2_eff * A * B - env.decay * C + env.inflow

    return np.array([dA, dB, dC], dtype=float)


def simulate_minimal_origin_life(
    t_max: float = 500.0,
    dt: float = 1.0,
    initial_state: np.ndarray = None,
    env: EnvironmentParams = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the minimal origin-of-life system with explicit Euler integration.

    Parameters
    ----------
    t_max : float
        Final simulation time.
    dt : float
        Time step.
    initial_state : np.ndarray
        Initial concentrations (A0, B0, C0). If None, defaults to (1,1,1).
    env : EnvironmentParams
        Environment and twist parameters. If None, defaults are used.

    Returns
    -------
    times : np.ndarray, shape (N,)
    states : np.ndarray, shape (N, 3)
        states[i] = [A(t_i), B(t_i), C(t_i)]
    """
    if env is None:
        env = EnvironmentParams()

    if initial_state is None:
        state = np.array([1.0, 1.0, 1.0], dtype=float)
    else:
        state = np.array(initial_state, dtype=float)

    n_steps = int(t_max / dt) + 1
    times = np.linspace(0.0, t_max, n_steps)
    states = np.zeros((n_steps, 3), dtype=float)
    states[0] = state

    for i in range(1, n_steps):
        t = times[i - 1]
        rhs = reaction_rhs(states[i - 1], t, env)
        states[i] = states[i - 1] + dt * rhs
        # ensure non-negative concentrations
        states[i] = np.maximum(states[i], 0.0)

    return times, states


def compute_sigma_lock_index(times: np.ndarray, states: np.ndarray) -> np.ndarray:
    """
    Compute a simple σ-locking index L(t) from the trajectory.

    Heuristic:
    - L is high if the system changes slowly (small |dx/dt|)
      and the three species are in balance (small std across A,B,C).
    - L is low if the system changes fast or is strongly unbalanced.

    Implementation:
    - numerical velocity v(t_i) = ||x_i - x_{i-1}|| / dt
    - balance measure b(t_i) = exp(-std(x_i) / s0)
    - stability measure s(t_i) = exp(-v(t_i) / v0)
    - L(t_i) = b(t_i) * s(t_i)

    Parameters
    ----------
    times : np.ndarray, shape (N,)
    states : np.ndarray, shape (N, 3)

    Returns
    -------
    L : np.ndarray, shape (N,)
    """
    dt = np.diff(times)
    # avoid division by zero
    dt[dt == 0] = 1.0

    # numerical "velocity"
    diffs = np.diff(states, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt

    # scaling factors
    v0 = np.percentile(speeds, 75) + 1e-9  # typical "fast" speed
    s0 = np.percentile(np.std(states, axis=1), 75) + 1e-9

    L = np.zeros_like(times, dtype=float)
    for i in range(1, len(times)):
        x = states[i]
        v = speeds[i - 1]
        balance = np.exp(-np.std(x) / s0)
        stability = np.exp(-v / v0)
        L[i] = balance * stability

    L[0] = L[1]  # simple initialization

    return L


def run_origin_life_example(
    output_dir: Union[str, Path],
    t_max: float = 500.0,
    dt: float = 1.0,
    env: EnvironmentParams = None,
) -> pd.DataFrame:
    """
    High-level pipeline for the origin-of-life example used in the paper.

    - Simulate minimal autocatalytic network with twist driving.
    - Compute σ-locking index L(t).
    - Save results as CSV in output_dir.
    - Return DataFrame with columns: time, A, B, C, L.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the CSV should be written.
    t_max : float
        Final simulation time.
    dt : float
        Time step.
    env : EnvironmentParams
        Environment parameters. If None, defaults are used.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with the simulation results.
    """
    if env is None:
        env = EnvironmentParams()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "minimal_sigma_lock_results.csv"

    times, states = simulate_minimal_origin_life(t_max=t_max, dt=dt, env=env)
    L = compute_sigma_lock_index(times, states)

    df = pd.DataFrame(
        {
            "time": times,
            "A": states[:, 0],
            "B": states[:, 1],
            "C": states[:, 2],
            "L": L,
        }
    )
    df.to_csv(out_file, index=False)
    return df
