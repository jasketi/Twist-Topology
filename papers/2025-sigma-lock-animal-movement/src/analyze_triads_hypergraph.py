#!/usr/bin/env python3
import pandas as pd
import itertools
import networkx as nx

TRIADS_FILE = "triads_scan.csv"
STRIAD_THR = 0.05   # nur Triaden mit Striad >= 0.05 verwenden (kannst du später anpassen)

print(f"1) Lese {TRIADS_FILE} ...")
df = pd.read_csv(TRIADS_FILE)
print("   Zeilen gesamt:", len(df))

# Optional: Triaden nach Stärke filtern
strong = df[df["Striad"] >= STRIAD_THR].copy()
print(f"2) Triaden mit Striad >= {STRIAD_THR:0.3f}: {len(strong)}")

# Alle beteiligten Störche sammeln
nodes = sorted(set(strong["A"]) | set(strong["B"]) | set(strong["C"]))
print("   Beteiligte Störche:", len(nodes))

G = nx.Graph()
G.add_nodes_from(nodes)

# Für jede Triade Kante zwischen allen Paaren (2-Schnitt des Hypergraphen)
for _, row in strong.iterrows():
    triple = [row["A"], row["B"], row["C"]]
    for u, v in itertools.combinations(triple, 2):
        G.add_edge(u, v)

print("\n3) Verbundene Komponenten im Triaden-Geflecht:")
components = list(nx.connected_components(G))
for k, comp in enumerate(components, 1):
    print(f"   Komponente {k} (|V| = {len(comp)}):")
    for s in sorted(comp):
        print("      -", s)

print("\n4) Kanten-Zusammenfassung:")
print("   Anzahl Knoten:", G.number_of_nodes())
print("   Anzahl Kanten:", G.number_of_edges())
