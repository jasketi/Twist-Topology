# Sigma-Twist in Animal Movement

This directory contains the analysis code and notebooks used for the paper

> Ascher, J. (20XX). *Sigma-Twist in Tierverhalten* (working title).

The **general σ-Lock / TwistX framework** and core algorithms live in the main
repository structure:

- Core algorithms: `core/` (σ-Lock, TwistX, triadic interactions, etc.)
- Project root: <https://github.com/jasketi/Twist-Topology>

This folder adds the **paper-specific parts** for the application to animal
movement (e.g. Movebank trajectories, episodic σ-locks, twist-topological
stability of collective behaviour).

## Directory layout

- `notebooks/` – Jupyter notebooks used to generate the figures and tables
- `src/`        – Python scripts and helper functions (preprocessing, analysis)
- `results/`    – small example output files (e.g. CSVs, PNGs), no large raw data

## Environment

A conda environment file is provided in `environment.yml`.

Typical usage:

```bash
cd path/to/Twist-Topology/papers/2025-sigma-lock-animal-movement

conda env create -f environment.yml
conda activate twist-animal-movement

jupyter lab

