# fBmChemotaxis

**fBmChemotaxis** is a Python-based simulation framework for chemotaxis models with autocorrelated noise. The code implements a simple stochastic model in which cell motility is driven by a combination of the gradient of a chemoattractant and fractional Brownian motion (fBm), formulated as a stochastic differential equation (SDE).

The framework supports simulation of fBm increments using both the **Davies–Harte method** and the **Cholesky method**, and advances the resulting SDEs using an Euler-type numerical scheme.

Two model variants are included:
- A **2D chemotaxis model on a manifold**, formulated in local coordinates when a global chart is available (`manifold.py`).
- A **1D interacting particle model**, incorporating diffusible, self-generated signals emitted by activated cells (`interacting.py`).

## Features
- Simulation of chemotaxis models with fractional Brownian motion.
- Support for 2D dynamics on general manifolds (or Euclidean space as a special case).
- Modeling of cell–cell communication via diffusible chemical signals.
- Multiple numerical methods for fBm generation.

## Installation

### Prerequisites
- Python 3
- `numpy`
- `matplotlib`

No additional dependencies are required.

## Usage

### 2D chemotaxis on a manifold
Run:
```bash
python manifold.py
```

To recover the Euclidean case, set the manifold parameters to zero in the corresponding configuration.

### 1D interacting chemotaxis model
Run:
```bash
python interacting.py
```

## File structure
- `manifold.py` – Simulation code for 2D chemotaxis on a manifold (or Euclidean space).
- `interacting.py` – Simulation code for 1D chemotaxis with interacting cells.
- `README.md` – Repository documentation.

## Citation
If you use **fBmChemotaxis** in your research, please cite:

G. Cornejo-Olea, L. Buvinic, J. Darbon, R. Erban, A. Ravasio, and A. Matzavinos.  *On the role of fractional Brownian motion in models of chemotaxis and stochastic gradient ascent.* Submitted, 2025. Preprint available on arXiv: **2511.18745**.
