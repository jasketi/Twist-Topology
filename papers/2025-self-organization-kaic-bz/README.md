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

## KaiC data (manual download)

Download the KaiC source data from the eLife article (DOI: 10.7554/eLife.23539) and place the relevant CSV/TSV under:
- `papers/2025-self-organization-kaic-bz/data/raw/kaiC/`
Then run (example):
```bash
python papers/2025-self-organization-kaic-bz/src/preprocess_kaiC.py --infile papers/2025-self-organization-kaic-bz/data/raw/kaiC/<FILE>.csv
```



### Fig S2 (hidden signals; BZ)

Run:

```bash
python papers/2025-self-organization-kaic-bz/src/make_figS2_hidden.py --config papers/2025-self-organization-kaic-bz/configs/quick.yaml --indir papers/2025-self-organization-kaic-bz/data/processed/bz/hidden --outdir papers/2025-self-organization-kaic-bz/results
```

Outputs:

- `results/figS2_hidden_PLI_matrix.png`
- `results/figS2_hidden_drift_matrix.png`
- `results/figS2_hidden_pairs_classified.csv`
- `results/figS2_hidden_summary.json`
- `results/figS2_hidden_top20_by_PLI.csv`

### Table 1 (KaiC pairwise phase metrics)

Run:

```bash
python papers/2025-self-organization-kaic-bz/src/make_table1_kaiC.py --config papers/2025-self-organization-kaic-bz/configs/quick.yaml
```

Outputs:

- `results/table1_kaiC.csv`
- `results/table1_kaiC_rows.tex`
- `results/run_meta_table1_kaiC.json`

### Fig 1 (BZ droplets summary)

Run:

```bash
python papers/2025-self-organization-kaic-bz/src/make_fig1_bz.py --config papers/2025-self-organization-kaic-bz/configs/quick.yaml
```

Outputs:

- `results/fig1_bz_winding_hist.png`
- `results/fig1_bz_period_hist.png`
- `results/fig1_bz_input_head.csv`
- `results/run_meta_fig1_bz.json`

### Fig 2 (KaiC summary + Kuramoto fits)

Run:

```bash
python papers/2025-self-organization-kaic-bz/src/make_fig2_kaiC.py --config papers/2025-self-organization-kaic-bz/configs/quick.yaml
```

Outputs:

- `results/fig2A_kaiC_period_vs_LD.png`
- `results/fig2B_kaiC_K_vs_Domega.png`
- `results/run_meta_fig2_kaiC.json`

## Processed inputs bundle (recommended for full reproduction)

For full reproduction on a fresh clone, download the processed inputs bundle (ZIP) from the project archive (e.g., Zenodo) and unpack it into:
- 

Then run:
```bash
python papers/2025-self-organization-kaic-bz/src/run_all.py --config papers/2025-self-organization-kaic-bz/configs/default.yaml
```

To verify integrity (SHA256), use:
```bash
python papers/2025-self-organization-kaic-bz/scripts/verify_inputs_bundle.py
```
