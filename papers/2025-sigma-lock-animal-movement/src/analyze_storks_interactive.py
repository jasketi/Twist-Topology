import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) CSVs einlesen
# ------------------------------------------------------------------
triad_roles = pd.read_csv("triad_roles_lock_all60.csv")
triads = pd.read_csv("triads_lock_all60.csv")

# ------------------------------------------------------------------
# 2) Dyadische Lock-Längen pro Paar aus triads_lock_all60.csv holen
# ------------------------------------------------------------------
def pairs_from(df, col1, col2, len_col):
    """Hilfsfunktion: hole Paare + Locklänge in ein einheitliches Format."""
    return df[[col1, col2, len_col]].rename(
        columns={col1: "i", col2: "j", len_col: "lock_len"}
    )

pairs_all = pd.concat(
    [
        pairs_from(triads, "A", "B", "lock_len_AB"),
        pairs_from(triads, "B", "C", "lock_len_BC"),
        pairs_from(triads, "C", "A", "lock_len_CA"),
    ],
    ignore_index=True,
)

# sortiertes Paar als Key, damit (A,B) und (B,A) identisch sind
pairs_all["pair"] = pairs_all.apply(
    lambda r: tuple(sorted((r["i"], r["j"]))), axis=1
)

# jedes Paar genau einmal (Lock-Länge ist über Triads konstant)
pair_lock = pairs_all.drop_duplicates(subset="pair")

# ------------------------------------------------------------------
# 3) Dyadische Zentralität pro Storch = Summe aller Lock-Längen über Partner
# ------------------------------------------------------------------
centrality = {}
for _, row in pair_lock.iterrows():
    i, j, L = row["i"], row["j"], row["lock_len"]
    centrality[i] = centrality.get(i, 0.0) + L
    centrality[j] = centrality.get(j, 0.0) + L

triad_roles["dyadic_centrality"] = triad_roles["stork"].map(centrality)

# ------------------------------------------------------------------
# 4) Z-Scores und Ränge für dyadisch vs. triadisch
# ------------------------------------------------------------------
triad_roles["z_dyad"] = (
    triad_roles["dyadic_centrality"]
    - triad_roles["dyadic_centrality"].mean()
) / triad_roles["dyadic_centrality"].std()

triad_roles["z_triad"] = (
    triad_roles["mean_S_triad_lock"]
    - triad_roles["mean_S_triad_lock"].mean()
) / triad_roles["mean_S_triad_lock"].std()

triad_roles["rank_triad"] = triad_roles["mean_S_triad_lock"].rank(
    ascending=False, method="min"
)
triad_roles["rank_dyad"] = triad_roles["dyadic_centrality"].rank(
    ascending=False, method="min"
)

# ------------------------------------------------------------------
# 5) Kandidaten für "interaktiv": hoch triadisch, unterdurchschnittlich dyadisch
# ------------------------------------------------------------------
interactive = triad_roles[
    (triad_roles["z_triad"] > 0.35) & (triad_roles["z_dyad"] < 0.0)
].sort_values("z_triad", ascending=False)

print("\nInteraktiv-Kandidaten (hoch triadisch, mittel/niedrig dyadisch):\n")
print(
    interactive[
        [
            "stork",
            "mean_S_triad_lock",
            "dyadic_centrality",
            "rank_triad",
            "rank_dyad",
            "z_triad",
            "z_dyad",
        ]
    ].to_string(index=False)
)

# Alles zur weiteren Verwendung abspeichern
triad_roles.to_csv("storks_triad_dyad_roles_all60.csv", index=False)
interactive.to_csv("storks_interactive_candidates.csv", index=False)

# ------------------------------------------------------------------
# 6) Scatter-Plot für die Paper-Grafik
# ------------------------------------------------------------------
plt.figure()
plt.scatter(triad_roles["z_dyad"], triad_roles["z_triad"])

# interaktive Störche beschriften
for _, r in interactive.iterrows():
    name_short = r["stork"].split(" / ")[0]
    plt.text(r["z_dyad"], r["z_triad"], name_short, fontsize=8)

plt.axvline(0, linestyle="--")
plt.axhline(0, linestyle="--")
plt.xlabel("Dyadische Lock-Zentralität (z)")
plt.ylabel("Triadische Stabilität mean_S_triad_lock (z)")
plt.tight_layout()
plt.savefig("fig_storks_dyad_vs_triad_roles_all60.pdf")

print("\nPlot gespeichert als fig_storks_dyad_vs_triad_roles_all60.pdf\n")

