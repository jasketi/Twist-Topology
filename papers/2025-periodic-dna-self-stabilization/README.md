# 2025-periodic-dna-self-stabilization

This folder contains the manuscript and a reproducible analysis of cross-correlation data from Patel et al. (2023),
supporting the paper:

**Periodic Self-Stabilization of DNA Activity Reveals a Geometric Feedback Between Structure and Function**
Jörg Ascher (Independent Researcher, Germany)

## Contents
- `paper/` — compiled paper PDF (`paper.pdf`)
- `figures/` — generated figures used in the manuscript
- `data/raw/` — 26 cross-correlation curves (*cross-correlation*.txt), see `SOURCE.md`
- `src/make_figures.py` — regenerates all figures and a derived summary table
- `outputs/fit_parameters.csv` — derived summary table (generated)

## Reproduce figures and outputs (one command)

From the repository root:

    python3 papers/2025-periodic-dna-self-stabilization/src/make_figures.py

Expected outputs:

    ls -lah papers/2025-periodic-dna-self-stabilization/figures \
           papers/2025-periodic-dna-self-stabilization/outputs

This will regenerate:
- `figures/fig_cis_trans.(png|pdf)`
- `figures/fig_modulators.(png|pdf)`
- `outputs/fit_parameters.csv`

## Data provenance
See `data/raw/SOURCE.md` for the upstream dataset reference and notes on duplicated basenames.

