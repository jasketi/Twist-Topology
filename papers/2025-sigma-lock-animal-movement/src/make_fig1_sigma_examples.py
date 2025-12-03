#!/usr/bin/env python3
"""
Figure 1: Examples of sigma-Lock detection in animal systems.

Zwei Beispiel-Dyaden:
- oben: Zeitserien von sigma_norm mit schattierten sigma-Lock-Episoden
- unten: Differential-Geschwindigkeitsraum (Δv_x, Δv_y) mit hervorgehobenen Lock-Punkten

Die Daten sind synthetisch, aber so gebaut, dass sie das Prinzip sauber illustrieren:
Dyad 1 mit wenigen längeren Locks, Dyad 2 mit mehreren kürzeren Locks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 8,
    "figure.dpi": 300,
})

rng = np.random.default_rng(42)

def make_sigma_and_state(T=900, dt=1.0, lock_windows=None):
    """
    Erzeugt synthetische sigma_norm-Zeitreihe und zugehörige
    Differential-Geschwindigkeiten (dvx, dvy).
    lock_windows: Liste von (t_start, t_end) in Sekunden.
    """
    t = np.arange(0, T, dt)

    # ----- sigma_norm oben: Basis + klare Lock-Buckel -----
    sigma = 0.1 + 0.05 * rng.standard_normal(len(t))
    thresh = 0.25

    lock_mask = np.zeros_like(t, dtype=bool)
    if lock_windows is not None:
        for (t0, t1) in lock_windows:
            mask = (t >= t0) & (t <= t1)
            lock_mask |= mask
            # innerhalb des Fensters sigma nach oben schieben
            sigma[mask] += 0.25 + 0.05 * rng.standard_normal(mask.sum())

    # ----- dvx / dvy unten: Werte im Bereich [-1, 1] -----
    dvx = rng.normal(scale=0.8, size=len(t))
    dvy = rng.normal(scale=0.8, size=len(t))

    # Lock-Fenster: kompakte Cluster (Kreise) im State-Space
    if lock_windows is not None:
        for (t0, t1) in lock_windows:
            mask = (t >= t0) & (t <= t1)
            n = mask.sum()
            if n > 0:
                theta = np.linspace(0, 2*np.pi, n, endpoint=False)
                radius = 0.5
                cx = rng.normal(scale=0.3)
                cy = rng.normal(scale=0.3)
                dvx[mask] = cx + radius * np.cos(theta) + 0.1 * rng.standard_normal(n)
                dvy[mask] = cy + radius * np.sin(theta) + 0.1 * rng.standard_normal(n)

    return t, sigma, thresh, lock_mask, dvx, dvy


def shade_lock_windows(ax, t_vals, lock_mask):
    """Schattiert zusammenhängende Blöcke von lock_mask auf Achse ax."""
    if not lock_mask.any():
        return
    idx = np.where(lock_mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, splits + 1)
    for block in blocks:
        ax.axvspan(t_vals[block[0]], t_vals[block[-1]],
                   color="0.85", alpha=0.6, zorder=0)


# Beispiel-Dyade 1: wenige längere Locks
t1, sigma1, thr1, lock1, dvx1, dvy1 = make_sigma_and_state(
    T=900,
    dt=1.0,
    lock_windows=[(120, 220), (420, 540)]
)

# Beispiel-Dyade 2: mehrere kürzere Locks
t2, sigma2, thr2, lock2, dvx2, dvy2 = make_sigma_and_state(
    T=900,
    dt=1.0,
    lock_windows=[(80, 130), (260, 310), (500, 560)]
)

# Zeit in Minuten für x-Achse (0 bis 15)
t1_min = t1 / 60.0
t2_min = t2 / 60.0

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex="col")

# --------- oben: sigma_norm Zeitserien (wie in der „guten“ Version) ---------
ax = axes[0, 0]
ax.plot(t1_min, sigma1, lw=1.0)
ax.axhline(thr1, color="0.4", linestyle="--", linewidth=0.8)
shade_lock_windows(ax, t1_min, lock1)
ax.set_ylabel(r"$\sigma^{\mathrm{norm}}$")
ax.set_title("example dyad 1 (e.g. bats)")

ax = axes[0, 1]
ax.plot(t2_min, sigma2, lw=1.0)
ax.axhline(thr2, color="0.4", linestyle="--", linewidth=0.8)
shade_lock_windows(ax, t2_min, lock2)
ax.set_title("example dyad 2 (e.g. birds)")

for ax in axes[0, :]:
    ax.set_xlim(t1_min[0], t1_min[-1])
    ax.set_xlabel("time (min)")

# --------- unten: Differential-velocity-Raum (Version, die du gut fandest) ---------
def plot_state_space(ax, dvx, dvy, lock_mask, show_legend=False):
    # 1/3 weniger Punkte: jedes 3. Sample nehmen
    dvx_sub = dvx[::3]
    dvy_sub = dvy[::3]
    lock_sub = lock_mask[::3]

    # outside-Lock: gut sichtbares Grau
    ax.scatter(dvx_sub[~lock_sub], dvy_sub[~lock_sub],
               s=10, alpha=0.5, color="0.4", label=r"outside $\sigma$-Lock")
    # inside-Lock: kräftiges Blau
    ax.scatter(dvx_sub[lock_sub], dvy_sub[lock_sub],
               s=16, alpha=0.9, color="tab:blue", label=r"inside $\sigma$-Lock")

    ax.set_xlabel(r"$\Delta v_x$")
    ax.set_ylabel(r"$\Delta v_y$")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal", adjustable="box")

    # wenige, lesbare Ticks
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    if show_legend:
        # Legende außerhalb des Rahmens, rechts
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=7,
            frameon=False,
        )

ax = axes[1, 0]
plot_state_space(ax, dvx1, dvy1, lock1, show_legend=True)
ax.set_title("differential velocity space")

ax = axes[1, 1]
plot_state_space(ax, dvx2, dvy2, lock2, show_legend=False)
ax.set_title("differential velocity space")

fig.tight_layout()

out_path = Path(__file__).with_name("fig_sigma_examples.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"✔ fig_sigma_examples.pdf geschrieben: {out_path}")
