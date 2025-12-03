# alten Inhalt löschen, Code oben reinkopieren, speichern
#!/usr/bin/env python3
"""
Erzeugt Figure 1 (fig_sigma_examples.pdf):
- Zeitreihen von sigma_norm für zwei Beispiel-Dyaden
- zugehörige Phasenraum-Plots im differentiellen Geschwindigkeitsraum (Δv_x, Δv_y)

Die Beispiele sind so konstruiert, dass:
- in BEIDEN Zeitreihen σ_norm mehrmals die Lock-Schwelle überschreitet
  (episodisches Rein/Raus),
- in BEIDEN Phasenraum-Panels Punkte sowohl "outside σ-Lock" als auch "inside σ-Lock" vorkommen.
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def intervals_from_mask(t, mask):
    """Finde zusammenhängende Intervalle in einem Bool-Array."""
    t = np.asarray(t)
    mask = np.asarray(mask, dtype=bool)
    intervals = []
    in_block = False
    start = None

    for i, flag in enumerate(mask):
        if flag and not in_block:
            in_block = True
            start = t[i]
        elif not flag and in_block:
            in_block = False
            end = t[i - 1]
            intervals.append((start, end))

    if in_block:
        intervals.append((start, t[-1]))

    return intervals


def make_lock_mask_from_sigma(t, sigma, theta, min_len=0.03):
    """
    Erzeuge einen Lock-Masken-Array:
    - Rohmaske: sigma > theta
    - Kleine Stücke kürzer als `min_len` werden verworfen.
    """
    raw = sigma > theta
    raw_intervals = intervals_from_mask(t, raw)
    long_intervals = [(s, e) for (s, e) in raw_intervals if (e - s) >= min_len]

    mask = np.zeros_like(t, dtype=bool)
    for s, e in long_intervals:
        mask |= (t >= s) & (t <= e)
    return mask


def make_example_data(t, theta):
    """
    Synthetische, aber „schön“ episodische Daten für zwei Dyaden.

    Beide σ_norm-Kurven kreuzen die Schwelle mehrfach, und beide Dyaden haben
    inside/outside-Punkte im differential velocity space.
    """
    x = t / t.max()

    # Dyade 1: eher "bat-like" – etwas unruhiger, kürzerer Lock
    rng1 = np.random.default_rng(1)
    sigma1 = 0.18 + 0.10 * np.sin(2 * np.pi * 1.1 * x + 0.1)
    sigma1 += 0.012 * rng1.standard_normal(len(t))
    sigma1 = np.clip(sigma1, 0.0, 0.5)
    lock_mask1 = make_lock_mask_from_sigma(t, sigma1, theta, min_len=0.03)

    # Dyade 2: eher "bird-like" – etwas glatter, längerer Lock
    rng2 = np.random.default_rng(2)
    sigma2 = 0.20 + 0.12 * np.sin(2 * np.pi * 0.9 * x - 0.6)
    sigma2 += 0.018 * rng2.standard_normal(len(t))
    sigma2 = np.clip(sigma2, 0.0, 0.5)
    lock_mask2 = make_lock_mask_from_sigma(t, sigma2, theta, min_len=0.03)

    n = len(t)
    rngv = np.random.default_rng(10)

    # Differential velocities: outside = breit; inside = enger um (0,0)
    dvx1 = rngv.normal(0, 1.0, n)
    dvy1 = rngv.normal(0, 1.0, n)
    dvx2 = rngv.normal(0, 1.0, n)
    dvy2 = rngv.normal(0, 1.0, n)

    for mask, dvx, dvy in [(lock_mask1, dvx1, dvy1),
                           (lock_mask2, dvx2, dvy2)]:
        if mask.any():
            dvx[mask] = rngv.normal(0, 0.4, mask.sum())
            dvy[mask] = rngv.normal(0, 0.4, mask.sum())

    return sigma1, sigma2, lock_mask1, lock_mask2, dvx1, dvy1, dvx2, dvy2


def plot_sigma_examples(out_path="figures/fig_sigma_examples.pdf"):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Zeitachse (0–0.4 min) wie in deinem Beispiel
    t = np.linspace(0.0, 0.4, 200)
    theta = 0.2

    (sigma1, sigma2,
     lock_mask1, lock_mask2,
     dvx1, dvy1, dvx2, dvy2) = make_example_data(t, theta)

    intervals1 = intervals_from_mask(t, lock_mask1)
    intervals2 = intervals_from_mask(t, lock_mask2)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(6.5, 4.0),
        gridspec_kw={"height_ratios": [1.0, 1.0]}
    )
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # ---------- obere Reihe: σ_norm über Zeit ----------
    # Dyad 1
    ax1.plot(t, sigma1, linewidth=1.0)
    ax1.axhline(theta, linestyle="--", linewidth=0.8)
    for s, e in intervals1:
        ax1.axvspan(s, e, color="0.9")
    ax1.set_ylabel(r"$\sigma_{\mathrm{norm}}$")
    ax1.set_xlabel("time (min)")
    ax1.set_title("example dyad 1 (e.g. bats)")
    ax1.set_xlim(0.0, 0.4)
    ax1.set_ylim(0.0, 0.5)
    ax1.set_xticks([0.0, 0.2, 0.4])
    ax1.set_yticks([0.0, 0.2, 0.4])

    # Dyad 2
    ax2.plot(t, sigma2, linewidth=1.0)
    ax2.axhline(theta, linestyle="--", linewidth=0.8)
    for s, e in intervals2:
        ax2.axvspan(s, e, color="0.9")
    ax2.set_xlabel("time (min)")
    ax2.set_title("example dyad 2 (e.g. birds)")
    ax2.set_xlim(0.0, 0.4)
    ax2.set_ylim(0.0, 0.5)
    ax2.set_xticks([0.0, 0.2, 0.4])
    ax2.set_yticks([0.0, 0.2, 0.4])
    ax2.set_yticklabels([])

    # ---------- untere Reihe: differential velocity space ----------
    # Dyad 1
    outside1 = ~lock_mask1
    ax3.scatter(dvx1[outside1], dvy1[outside1],
                s=14, alpha=0.5, label="outside $\\sigma$-Lock")
    ax3.scatter(dvx1[lock_mask1], dvy1[lock_mask1],
                s=14, alpha=0.9, label="inside $\\sigma$-Lock")
    ax3.set_xlabel(r"$\Delta v_x$")
    ax3.set_ylabel(r"$\Delta v_y$")
    ax3.set_title("differential velocity space")
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_xticks([-2, -1, 0, 1, 2])
    ax3.set_yticks([-2, -1, 0, 1, 2])

    # Dyad 2
    outside2 = ~lock_mask2
    ax4.scatter(dvx2[outside2], dvy2[outside2],
                s=14, alpha=0.5, label="outside $\\sigma$-Lock")
    ax4.scatter(dvx2[lock_mask2], dvy2[lock_mask2],
                s=14, alpha=0.9, label="inside $\\sigma$-Lock")
    ax4.set_xlabel(r"$\Delta v_x$")
    ax4.set_ylabel(r"$\Delta v_y$")
    ax4.set_title("differential velocity space")
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_xticks([-2, -1, 0, 1, 2])
    ax4.set_yticks([-2, -1, 0, 1, 2])

    # Legende (einmal unten zentriert)
    handles, labels = ax4.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=2,
               frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote figure to {out_path.resolve()}")


if __name__ == "__main__":
    plot_sigma_examples()

