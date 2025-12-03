#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Datensätze (kannst du später anpassen/erweitern)
datasets = [
    "Mexican Myotis",
    "P. hastatus",
    "P. lylei",
    "E. helvum (Africa)",
    "E. helvum (seed)",
    "Free-tailed bats",
    "Griffon vultures",
    "White storks",
]

# Dyadischer Index D = Anteil signifikanter σ-Locks (Nsig/Npairs)
# --> hier erstmal grob die Werte aus Table 1 / unseren Analysen
D = [
    0.40,   # Mexican Myotis
    0.058,  # P. hastatus
    0.063,  # P. lylei
    0.194,  # E. helvum Africa
    0.032,  # E. helvum seed
    0.00,   # Free-tailed (keine signifikanten Locks)
    0.088,  # Griffons (8 von 91 Dyaden ~ 0.088)
    0.25,   # White storks (Platzhalter, kannst du später schärfen)
]

# Triadischer Index T = "Stärke" triadischer Organisation
# Hier vorerst qualitativ normiert (0 = keine, 1 = sehr stark).
# Du kannst das später durch echte Zahlen ersetzen (z.B. Anteil signifikanter Triads).
T = [
    0.00,  # Mexican Myotis: praktisch keine Triaden
    0.05,  # P. hastatus: kaum / kurze Triads
    0.10,  # P. lylei: kurze Triplets
    0.60,  # E. helvum Africa: viele multi-hour Triads im Kern
    0.50,  # E. helvum seed: 4-5 Bat-Konvoi mit starken Triads
    0.10,  # Free-tailed: viele kurze, flackernde Triplets
    0.70,  # Griffons: Konvoi-Kern, viele Triads mit Stunden-Lockzeit
    1.00,  # White storks: alle 56 Triads hoch signifikant
]

plt.figure()
plt.scatter(D, T)

for x, y, lab in zip(D, T, datasets):
    plt.text(x, y, lab, fontsize=8, ha="left", va="bottom")

plt.xlabel("Dyadic index $D$ (fraction of significant $\\sigma$-Locks)")
plt.ylabel("Triadic index $T$ (triadic organisation strength)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_dyad_triad_phase.pdf")
print("FERTIG: fig_dyad_triad_phase.pdf geschrieben.")
