# sigma-lock-ews

Reproduction package for the manuscript:

**“Sigma-Lock and Twist Topology in Dynamical Systems: From Mandelbrot and Logistic to Lorenz, Gaia and Early-Warning Signals”**  
Jörg Ascher (Independent Researcher, Germany)

## Folder layout

- Paper folder (this): `papers/sigma-lock-ews/`
- Importable python wrapper (no hyphen): `papers/sigma_lock_ews/`
- Generic algorithms: `twistlab/`

The paper references `papers/sigma-lock-ews` as the location for paper-specific scripts/notebooks.

## Quickstart (smoke test)

From repository root:

```bash
python -m papers.sigma_lock_ews.src.run_lorenz_ews --config papers/sigma-lock-ews/configs/quick.yaml
```

Outputs:

- `papers/sigma-lock-ews/results/run_meta.json`
- `papers/sigma-lock-ews/results/PLACEHOLDER.txt`

## Full reproduction

```bash
python -m papers.sigma_lock_ews.src.run_lorenz_ews --config papers/sigma-lock-ews/configs/default.yaml
```

## Environment

```bash
conda env create -f papers/sigma-lock-ews/environment.yml
conda activate <ENVNAME>
pip install -e .
```

## TODO

- Connect `run_lorenz_ews.py` to the real `twistlab` pipeline and write paper figures/tables into `papers/sigma-lock-ews/results/`.


## Reproduction targets

### Table 1 (Lorenz ramp sensitivity; Δρ lead)

Run (full/default):

```bash
python papers/sigma-lock-ews/src/make_table1_lorenz.py --config papers/sigma-lock-ews/configs/default.yaml --outdir papers/sigma-lock-ews/results
```

Outputs:

- `papers/sigma-lock-ews/results/table1_lorenz.csv`
- `papers/sigma-lock-ews/results/table1_lorenz_rows.tex`
- `papers/sigma-lock-ews/results/run_meta_table1_lorenz.json`

### Smoke test (pipeline scaffold)

```bash
python -m papers.sigma_lock_ews.src.run_lorenz_ews --config papers/sigma-lock-ews/configs/quick.yaml
```

Outputs:

- `papers/sigma-lock-ews/results/run_meta.json`
- `papers/sigma-lock-ews/results/PLACEHOLDER.txt`
