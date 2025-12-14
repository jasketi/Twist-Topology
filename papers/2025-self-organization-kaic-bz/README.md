# 2025-self-organization-kaic-bz

Reproduction bundle for the manuscript:
**Toward a Twist-Topological Account of Self-Organization: From KaiC to the Belousov–Zhabotinsky Reaction**

## Folder layout
- configs/: quick vs default paper parameters
- src/: paper-specific scripts (reproduce figures/tables)
- data/: local raw/processed data (not committed)
- results/: generated outputs (not committed)

## Quickstart (scaffold)
From repo root:
```bash
python papers/2025-self-organization-kaic-bz/src/run_all.py --config papers/2025-self-organization-kaic-bz/configs/quick.yaml
```

## Reproduction targets
### Table skeletons + run meta (now)
- results/run_meta.json
- results/tables_skeleton.csv
- results/targets.txt

### Next wiring steps (TODO)
- fetch + preprocess KaiC dataset
- fetch + preprocess BZ dataset (SOTON/D0363)
- compute W, PLI, dW/dt and regenerate Fig/Table outputs from real data
