# Twist-Topological Constraints on the Origin of Life

This folder contains reproducibility materials for the manuscript:

> **"Twist-Topological Constraints on the Origin of Life:
> Why Life Cannot Be Causally Constructed but Only Probabilistically Enabled"**

- Preprint / published version: DOI: `10.xxxx/zenodo.xxxxxx`
- Repository: https://github.com/jasketi/Twist-Topology
- Contact: `deine_mail@…` (optional)

## Contents

- `notebooks/` – Jupyter notebooks for:
  - minimal autocatalytic network with twist-structured updates;
  - detection of σ-Lock events in simulation time series;
  - estimation of the σ-Lock rate λ_{σ-Lock}(E) and parameter sweeps.
- `figures/` – exported figures used in the paper (PDF/PNG).
- `data/` (optional) – raw or processed data used in the examples, if needed.

## Getting started

1. Clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/jasketi/Twist-Topology.git
   cd Twist-Topology
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

