σ-Locks in Molecules: Reproducibility
=====================================

This folder contains reproducibility materials for the manuscript
“σ-Locks in Molecules: A Twist-Topological View on Chemical Bonding and Stability”
(Ascher, 2025).

Contents
--------

- notebooks/    Jupyter notebooks used to generate figures and toy calculations
- figures/      exported PNG/PDF figure files
- README.md     this file

Running the notebooks
---------------------

1. Install twistlab from the repository root:

   git clone https://github.com/jasketi/Twist-Topology.git
   cd Twist-Topology
   pip install -e .

2. Change into this directory:

   cd papers/2025-sigma-locks-molecules

3. Start Jupyter and run the notebooks:

   jupyter lab

All calculations in this folder are lightweight examples and should
reproduce exactly (up to plotting aesthetics).


Scripts
-------

The following Python scripts in this folder reproduce the simple quantitative
examples in the manuscript:

- `h2_potential_and_loop.py`  
  Generates the three-panel H₂ figure (Figure 1) and writes
  `figures/H2_twist_figure.png`.

- `cc_sigma_proxy.py`  
  Computes the vibrational proxy σ_mol^(CC) for the C–C, C=C and C≡C
  bonds used to construct Table 1 and prints the results to the console.


## DOI

The archived version of the manuscript is available at:

**https://doi.org/10.5281/zenodo.17832572**

