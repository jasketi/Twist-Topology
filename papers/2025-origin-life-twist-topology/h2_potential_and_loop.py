#!/usr/bin/env python3
"""
Generate Figure 1 for the σ-locks-in-molecules paper:
(A) Morse-like potential E(R)
(B) same potential with a few vibrational levels
(C) state-space loop in (R, dR/dt) as minimal σ-lock
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Ensure output directory exists
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    # Panel A/B: Morse-like potential
    R = np.linspace(0.4, 2.0, 400)
    De = 1.0
    a = 2.0
    Re = 0.74
    V = De * (1.0 - np.exp(-a * (R - Re)))**2

    # Create figure with 3 horizontal panels
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Common style
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # (A) Potential
    axes[0].plot(R, V, linewidth=2)
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("E(R)")
    axes[0].set_title("(A) Potential")
    axes[0].set_xlim(0.4, 2.0)

    # (B) Potential + schematic vibrational levels
    axes[1].plot(R, V, linewidth=2)
    levels = [0.15, 0.35, 0.55, 0.7]
    for E in levels:
        axes[1].hlines(E, 0.5, 1.4, linestyles="dashed", linewidth=1)
    axes[1].set_xlabel("R")
    axes[1].set_title("(B) Vibrational levels")
    axes[1].set_xlim(0.4, 2.0)
    axes[1].set_yticks([])
    axes[1].set_ylabel("")

    # (C) State-space loop in (R, dR/dt)
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    A = 0.1
    omega = 1.0
    R_loop = Re + A * np.cos(t)
    dR_dt = -A * omega * np.sin(t)

    axes[2].plot(R_loop, dR_dt, linewidth=2)
    axes[2].set_xlabel("R")
    axes[2].set_ylabel("dR/dt")
    axes[2].set_title("(C) State-space loop")
    axes[2].set_xlim(Re - 0.2, Re + 0.2)

    fig.tight_layout()

    out_path = os.path.join(out_dir, "H2_twist_figure.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
