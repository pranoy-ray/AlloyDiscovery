# Electronic manifolds for extrapolative alloy discovery

[![Paper](https://img.shields.io/badge/Digital%20Discovery-10.1039%2FD6DD00105J-blue)](https://doi.org/10.1039/D6DD00105J)
[![arXiv](https://img.shields.io/badge/arXiv-2603.06953-b31b1b)](https://arxiv.org/abs/2603.06953)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the codebase for the framework presented in **"Electronic manifolds for extrapolative alloy discovery"** (RSC *Digital Discovery*, 2026). It is a computationally efficient, data-efficient pipeline for high-throughput screening of refractory high-entropy alloys (HEAs) that replaces expensive self-consistent DFT with the **non-interacting (pseudo) electron density** as the primary structural descriptor.

## Overview

The method builds a low-dimensional "electronic manifold" of the alloy design space and couples it with Bayesian active learning:

1. **Pseudo-density descriptor** — superposes isolated valence electron densities on the solid-solution lattice, avoiding self-consistent DFT.
2. **Feature engineering** — extracts directionally resolved **two-point spatial correlations (2PS)** from the density fields.
3. **Dimensionality reduction** — compresses the features via **Principal Component Analysis (PCA)** into the electronic manifold.
4. **Active learning** — trains a **Gaussian Process Regression (GPR)** surrogate with Bayesian active sampling for property prediction and extrapolation.

Reported results from the paper:

- **NMAE < 2%** for the bulk modulus of Al–Nb–Ti–Zr alloys using only **10 training samples** (<0.2% of the dataset).
- A transferable electronic packing manifold across the refractory BCC alloy family.
- **Zero-shot extrapolation** to a 7-component system (Mo–Nb–Ta–Ti–V–W–Zr) containing four elements absent from training.
- **Few-shot** predictions at **NMAE < 3%** for 7-component alloys after augmenting with just **20** target-domain samples.

## Directory Structure

```
AlloyDiscovery/
├── alloy_discovery/            # Main Python package
│   ├── feature_engineering.py  # Pseudo-densities → 2-point spatial correlations (2PS)
│   ├── pca_analysis.py         # PCA dimensionality reduction of the electronic manifold
│   └── gpr_extrapolation.py    # Bayesian active learning + GPR (4→7 component zero-/few-shot)
├── data/
│   └── psp8/                   # ONCV pseudopotential files (.psp8)
├── setup.py
├── LICENSE
└── README.md
```

`data/psp8/` is designated for optimized norm-conserving Vanderbilt (ONCV) pseudopotential files used to construct the non-interacting densities.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/pranoy-ray/AlloyDiscovery.git
cd AlloyDiscovery
pip install -e .
```

### Dependencies

Installed automatically via `setup.py`: `numpy`, `pandas`, `scipy`, `scikit-learn`, `h5py`, `torch`, `gpytorch`, `botorch`, `ase`, `pyvista`, `matplotlib`.

A GPU-enabled build of `torch` is recommended for the GPR/active-learning workflow.

## Usage

The package exposes three stages that map to the pipeline above:

```python
from alloy_discovery import feature_engineering   # pseudo-density → 2PS features
from alloy_discovery import pca_analysis           # PCA → electronic manifold
from alloy_discovery import gpr_extrapolation      # Bayesian active learning + GPR
```

Run the modules in order: compute 2PS features from the pseudo-densities, reduce them with PCA to obtain the electronic manifold, then fit the GPR surrogate and perform active-learning-driven prediction/extrapolation.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{ray2026electronicmanifolds,
  title   = {Electronic manifolds for extrapolative alloy discovery},
  author  = {Ray, Pranoy and Bhowmik, Sayan and Suryanarayana, Phanish and Kalidindi, Surya R. and Medford, Andrew J.},
  journal = {Digital Discovery},
  year    = {2026},
  doi     = {10.1039/D6DD00105J}
}
```

- Journal (RSC *Digital Discovery*): https://doi.org/10.1039/D6DD00105J
- Preprint (arXiv): https://arxiv.org/abs/2603.06953

## License

Released under the [MIT License](LICENSE).
