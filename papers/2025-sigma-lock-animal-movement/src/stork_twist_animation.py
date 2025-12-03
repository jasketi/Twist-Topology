#!/usr/bin/env python3
"""
Erzeugt ein GIF 'stork_trajectories.gif', das drei "Störche" (Leader, Follower, Interaktor)
in einer Thermik-Trajektorie zeigt. Die Trajektorien schließen sich zu Schleifen, und
gleichzeitig wird markiert, wann die Triade annähernd "gelockt" ist (dreieckige Schleifenschließung).

Visualisierung:
- Trajektorien aller drei Vögel
- aktuelle Positionen als Punkte
- dyadischer Lock:
    * eine Linie verbindet in jedem Frame das jeweils stärkste dyadische Paar (geringste Distanz)
- Triaden-Lock:
    * wenn Lock aktiv ist, verbindet ein Dreieck alle drei Vögel
    * Text "LOCK – Triaden-Schleife geschlossen"

Ausführen:
    python3 stork_twist_animation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # kein GUI-Backend nötig
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# -----------------------------
# 1. Parameter & Trajektorien
# -----------------------------

n_frames = 400         # Anzahl der Frames im GIF
dt = 0.05              # Zeitabstand zwischen Frames
t = np.linspace(0, n_frames * dt, n_frames)

# "Thermik-Zentrum" des Schwarms – bewegt sich leicht, um den Twist zu zeigen
cx = 0.3 * np.sin(0.5 * t)
cy = 0.3 * np.cos(0.5 * t)

# Grundradius (Abstand der Vögel vom Zentrum), leicht oszillierend
R_base = 1.0 + 0.2 * np.sin(0.3 * t)

# Grund-Winkelgeschwindigkeit (sie kreisen um die Thermik)
omega = 1.2
theta = omega * t

# Phasenverschiebungen für Leader, Follower, Interaktor
phase_offsets = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])

# Positionen der drei Vögel: shape = (3 Vögel, n_frames, 2 Koordinaten)
positions = np.zeros((3, n_frames, 2))

for b in range(3):
    # Jeder Vogel bekommt eine leichte zusätzliche Winkel-Modulation
    th = theta + phase_offsets[b] + 0.3 * np.sin(0.7 * t + b)
    # Und eine kleine Radialmodulation (damit es organischer aussieht)
    r = R_base * (1 + 0.05 * np.sin(0.9 * t + b))

    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)

    positions[b, :, 0] = x
    positions[b, :, 1] = y


# ----------------------------------------------------
# 2. "Lock"-Detektion: Wann schließt sich die Triade?
# ----------------------------------------------------

# Idee: Ein "Lock"-Frame ist einer, in dem die drei Vögel annähernd
# ein gleichseitiges Dreieck bilden (alle Abstände ähnlich).
n = n_frames
lock_mask = np.zeros(n, dtype=bool)

for i in range(n):
    p0, p1, p2 = positions[:, i, :]

    d01 = np.linalg.norm(p0 - p1)
    d12 = np.linalg.norm(p1 - p2)
    d20 = np.linalg.norm(p2 - p0)

    d_mean = (d01 + d12 + d20) / 3.0
    if d_mean == 0:
        continue

    # Toleranz: max. 8% Abweichung vom Mittelwert
    if max(abs(d01 - d_mean), abs(d12 - d_mean), abs(d20 - d_mean)) < 0.08 * d_mean:
        lock_mask[i] = True

# Glätten: wir wollen eher kurze Sequenzen als zusammenhängende Lock-Phasen sehen
window = 7
smoothed = np.convolve(lock_mask.astype(int), np.ones(window), mode="same")
lock_mask = smoothed >= (window - 1)


# -----------------------------
# 3. Plot & Animation
# -----------------------------

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.set_title("Storchen-Triade – Trajektorien & Schleifenschließungen")
ax.axis("off")

names = ["Leader", "Follower", "Interaktor"]
lines = []
scatters = []

for b in range(3):
    # Linien (Trajektorien)
    line, = ax.plot([], [], lw=1.5, label=names[b])
    # Punkte (aktuelle Position)
    scat = ax.scatter([], [])
    lines.append(line)
    scatters.append(scat)

ax.legend(loc="lower right")

# Text für Lock-Anzeige und Zeit
lock_text = ax.text(
    0.5, 0.95, "", ha="center", va="top",
    transform=ax.transAxes, fontsize=12, fontweight="bold"
)
time_text = ax.text(
    0.02, 0.02, "", ha="left", va="bottom",
    transform=ax.transAxes, fontsize=9
)

# Dreieck-Linie für Triaden-Locks (feiner)
triangle_line, = ax.plot([], [], lw=1.0)

# Linie für dyadischen Lock (stärkstes Paar im aktuellen Frame)
dyad_line, = ax.plot([], [], lw=0.8, linestyle="--")


def init():
    """Initialzustand der Animation."""
    for line in lines:
        line.set_data([], [])
    for scat in scatters:
        scat.set_offsets(np.zeros((1, 2)))
    lock_text.set_text("")
    time_text.set_text("")
    triangle_line.set_data([], [])
    dyad_line.set_data([], [])
    return lines + scatters + [lock_text, time_text, triangle_line, dyad_line]


def update(frame):
    """Update-Funktion für jeden Frame."""
    # Trajektorien bis zum aktuellen Zeitpunkt
    for b in range(3):
        lines[b].set_data(
            positions[b, :frame + 1, 0],
            positions[b, :frame + 1, 1],
        )
        scatters[b].set_offsets(positions[b, frame, :])

    # aktuelle Positionen
    p0 = positions[0, frame, :]
    p1 = positions[1, frame, :]
    p2 = positions[2, frame, :]

    # -------------------------
    # Dyadischer Lock: stärkstes Paar
    # -------------------------
    d01 = np.linalg.norm(p0 - p1)
    d12 = np.linalg.norm(p1 - p2)
    d20 = np.linalg.norm(p2 - p0)

    distances = np.array([d01, d12, d20])
    pair_idx = np.argmin(distances)  # Index des engsten Paars

    if pair_idx == 0:
        pa, pb = p0, p1
    elif pair_idx == 1:
        pa, pb = p1, p2
    else:
        pa, pb = p2, p0

    dyad_line.set_data([pa[0], pb[0]], [pa[1], pb[1]])

    # -------------------------
    # Triaden-Lock: geschlossenes Dreieck
    # -------------------------
    if lock_mask[frame]:
        lock_text.set_text("LOCK – Triaden-Schleife geschlossen")

        tri_x = [p0[0], p1[0], p2[0], p0[0]]
        tri_y = [p0[1], p1[1], p2[1], p0[1]]

        triangle_line.set_data(tri_x, tri_y)
    else:
        lock_text.set_text("")
        triangle_line.set_data([], [])

    time_text.set_text(f"t = {t[frame]:.1f}")
    return lines + scatters + [lock_text, time_text, triangle_line, dyad_line]


ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    blit=True
)

# GIF schreiben
output_file = "stork_trajectories.gif"
ani.save(output_file, writer=PillowWriter(fps=25))
print(f"Fertig! GIF gespeichert als: {output_file}")

