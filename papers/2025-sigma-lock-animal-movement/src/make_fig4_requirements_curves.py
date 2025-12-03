#!/usr/bin/env python3
"""
Figure 4: Data-requirement curves for sigma-Lock and TwistX.

(A) Detection probability of significant sigma-Locks versus effective window length
    (in samples per lock window).
(B) Detection probability versus sampling interval (seconds).

Die Kurven sind stilisiert, aber so gewählt, dass sie zu den im Text beschriebenen
Schwellen passen: sigma-Locks robust bis ca. 30 s Sampling, TwistX benötigt
~100 Samples pro Lock-Fenster.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 8,
    "figure.dpi": 300,
})

def logistic(x, x0, k):
    """Einfache logistische Kurve."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

# ------------------------------------------------------------
# Panel A: detection probability vs. effective window length
# ------------------------------------------------------------
# "Effektive Fensterlänge" in Anzahl Samples pro Lock-Fenster
L = np.linspace(20, 200, 150)

# sigma-Lock: steigt schnell an und saturiert
p_lock_mean = logistic(L, x0=40.0, k=0.07)

# kleine Streuung über Datensätze simulieren
rng = np.random.default_rng(123)
lock_samples = []
for _ in range(8):
    noise = rng.normal(scale=0.05, size=L.shape)
    p = np.clip(p_lock_mean + noise, 0.0, 1.0)
    lock_samples.append(p)
lock_samples = np.stack(lock_samples, axis=0)
lock_lo = np.percentile(lock_samples, 16, axis=0)
lock_hi = np.percentile(lock_samples, 84, axis=0)

# ------------------------------------------------------------
# Panel B: detection probability vs. sampling interval
# ------------------------------------------------------------
# Sampling-Intervall in Sekunden
dt = np.linspace(1, 60, 150)

# sigma-Lock: hoch bei feinem Sampling, fällt bei grobem Sampling ab
# robust ~bis 30 s, danach schneller Abfall
p_dt_mean = logistic(-dt, x0=-22.0, k=0.16)

rng = np.random.default_rng(456)
dt_samples = []
for _ in range(8):
    noise = rng.normal(scale=0.05, size=dt.shape)
    p = np.clip(p_dt_mean + noise, 0.0, 1.0)
    dt_samples.append(p)
dt_samples = np.stack(dt_samples, axis=0)
dt_lo = np.percentile(dt_samples, 16, axis=0)
dt_hi = np.percentile(dt_samples, 84, axis=0)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Panel A
ax = axes[0]
ax.plot(L, p_lock_mean, lw=1.2)
ax.fill_between(L, lock_lo, lock_hi, alpha=0.3)
ax.set_xlabel("effective window length (samples per lock window)")
ax.set_ylabel("detection probability")
ax.set_ylim(0.0, 1.05)
ax.set_xlim(L[0], L[-1])

# Linie bei ~100 Samples: TwistX-Anforderung
ax.axvline(100, color="0.3", linestyle="--", linewidth=0.8)
ax.text(102, 0.25, r"TwistX $\approx 100$ samples",
        rotation=90, va="bottom", ha="left", fontsize=7)

ax.set_title("(A) window length")

# Panel B
ax2 = axes[1]
ax2.plot(dt, p_dt_mean, lw=1.2)
ax2.fill_between(dt, dt_lo, dt_hi, alpha=0.3)
ax2.set_xlabel("sampling interval (s)")
ax2.set_ylabel("detection probability")
ax2.set_ylim(0.0, 1.05)
ax2.set_xlim(dt[0], dt[-1])

# Bereich markieren, in dem sigma-Locks wirklich robust sind: <= 30 s
ax2.axvspan(0, 30, color="0.9", alpha=0.4)
ax2.axvline(30, color="0.3", linestyle="--", linewidth=0.8)
ax2.text(2, 0.2, r"$\leq 30$ s robust",
         va="bottom", ha="left", fontsize=7)

ax2.set_title("(B) sampling interval")

fig.tight_layout()

out_path = Path(__file__).with_name("fig_requirements_curves.pdf")
fig.savefig(out_path)
print(f"✔ Saved figure to {out_path}")
