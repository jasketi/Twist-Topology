import pandas as pd
import matplotlib.pyplot as plt

# Einheitliche Schriftgrößen
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ---------- Daten laden ----------
roles = pd.read_csv("triad_roles_lock_all60.csv")
roles_60 = pd.read_csv("storks_triad_dyad_roles_all60.csv")

# Kurz-Namen extrahieren (alles vor " / ")
for df in (roles, roles_60):
    df["name_short"] = df["stork"].str.split(" / ").str[0]

# =========================================================
# Fig. 5A: 8 fokale Störche (Balou, Bello, Betty, Beringa,
#          Bubbel, Chompy, Conchito, Cookie)
# =========================================================
focal = ["Balou", "Bello", "Betty", "Beringa",
         "Bubbel", "Chompy", "Conchito", "Cookie"]

df8 = roles[roles["name_short"].isin(focal)].copy()
df8 = df8.sort_values("mean_S_triad_lock")

fig, ax = plt.subplots(figsize=(4, 3))

ax.barh(df8["name_short"], df8["mean_S_triad_lock"])

ax.set_xlabel(r"Mean triad stability $S_{\mathrm{triad}}$")
ax.set_ylabel("")

# n_triads ist für alle 8 gleich (=1711), das würde nur die Grafik
# vollschreiben, ohne Mehrwert. Deshalb KEINE Zahlen an den Balken.

plt.tight_layout()
fig.savefig("fig5A_triad_roles_new.pdf")
fig.savefig("fig5A_triad_roles_new.png", dpi=300)
plt.close(fig)

# =========================================================
# Fig. 5B: Rollenraum D vs. S_triad,lock
#          - extreme Outlier mit sehr kleinem z_triad werden
#            für die Darstellung ausgeblendet
#          - Legende statt Textlabels
# =========================================================

fig, ax = plt.subplots(figsize=(4, 4))

# Outlier raus: alles mit z_triad <= -2.0 wird nicht geplottet
filtered = roles_60[roles_60["z_triad"] > -2.0].copy()

# alle "normalen" Störche in hell
ax.scatter(filtered["z_dyad"], filtered["z_triad"], alpha=0.4)

triad_hubs = ["Cookie", "Bubbel", "Balou", "Conchito"]
interactive_hubs = ["Snowy", "Kim", "Peaches", "Zeffro"]

handles = []
labels = []

# Triad hubs (Kreise)
for name in triad_hubs:
    row = roles_60[roles_60["name_short"] == name].iloc[0]
    sc = ax.scatter(row["z_dyad"], row["z_triad"], marker="o")
    handles.append(sc)
    labels.append(name)

# Interactive hubs (Quadrate)
for name in interactive_hubs:
    row = roles_60[roles_60["name_short"] == name].iloc[0]
    sc = ax.scatter(row["z_dyad"], row["z_triad"], marker="s")
    handles.append(sc)
    labels.append(name)

# Null-Linien für z-Scores
ax.axvline(0, linestyle="--")
ax.axhline(0, linestyle="--")

# Achsen: an gefilterte Daten angepasst, mit etwas Rand
x_min = filtered["z_dyad"].min() - 0.3
x_max = filtered["z_dyad"].max() + 0.3
y_min = filtered["z_triad"].min() - 0.3
y_max = filtered["z_triad"].max() + 0.3
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("Dyadic lock centrality (z-score)")
ax.set_ylabel(r"Mean lock-based triad stability $S_{\mathrm{triad,lock}}$ (z-score)")

# Legende mit allen 8 hervorgehobenen Störchen,
# rechts außerhalb der Achse
ax.legend(handles, labels, title="Highlighted storks",
          loc="center left", bbox_to_anchor=(1.02, 0.5),
          borderaxespad=0.)

plt.tight_layout()
fig.savefig("fig5B_dyad_vs_triad_new.pdf", bbox_inches="tight")
fig.savefig("fig5B_dyad_vs_triad_new.png", dpi=300, bbox_inches="tight")
plt.close(fig)
