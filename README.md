# Alloy Discovery

This repository contains the codebase for the framework presented in the paper **"Universal electronic manifolds for extrapolative alloy discovery"**. It utilizes non-interacting pseudo-electron densities as structural descriptors for high-throughput High-Entropy Alloy (HEA) screening.

## Directory Structure
- `alloy_discovery/`: Main python package.
  - `feature_engineering.py`: Processes pseudo-densities and computes 2-Point Spatial Correlations (2PS).
  - `pca_analysis.py`: Handles Principal Component Analysis (PCA) for dimensionality reduction of the electronic manifold.
  - `gpr_extrapolation.py`: Implements Bayesian Active Learning and Gaussian Process Regression (GPR) for 4-to-7 component zero-shot/few-shot extrapolation.
- `data/psp8/`: Directory designated for optimized norm-conserving Vanderbilt (ONCV) pseudopotential files.

## Installation
Clone the repository and install the package locally:
```bash
git clone [https://github.com/your-username/AlloyDiscovery.git](https://github.com/your-username/AlloyDiscovery.git)
cd AlloyDiscovery
pip install -e .
