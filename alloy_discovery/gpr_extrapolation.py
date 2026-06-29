"""
gpr_extrapolation.py
Pranoy — GPR active-learning pipeline for multi-component RHEA alloys.

Changes from reviewer response:
  - ErrorMetrics.nmae: denominator is now np.mean(y_true) (mean-normalised),
    replacing the earlier range-normalised definition.
  - BayesianExperimentDesign: train_indices, X_train, y_train are persistent
    instance attributes so the object can be reset and reused across reviewer
    ablation experiments (Comment 1, 3) without re-instantiation.
  - BayesianExperimentDesign.history now tracks "NMAE" alongside MAPE/MAE/RMSE/R2.
  - RandomSamplingBaseline.history likewise tracks "NMAE".
  - New utility: nmae_by_quartile — per-quartile NMAE over an index subset.
  - New utility: unseen_fraction — summed Mo+Ta+V+W mole fraction for a formula.
  - New utility: dominant_elements — checks if a given element set exceeds a
    mole-fraction threshold (used in D4 cross-composition ablation, Comment 1).
"""

import os
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LCMKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Error metrics
# ============================================================

class ErrorMetrics:
    """Regression error metrics used throughout the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """MAPE with epsilon stabilisation for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        """
        Normalised MAE — normalised by the mean of y_true (not the range).
        Consistent with the reviewer-response notebook where NMAE is reported
        as |MAE / mean(y_true)| * 100 %.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE":       ErrorMetrics.mae(y_true, y_pred),
            "MAPE":      ErrorMetrics.mape(y_true, y_pred),
            "RMSE":      ErrorMetrics.rmse(y_true, y_pred),
            "NMAE":      ErrorMetrics.nmae(y_true, y_pred),
            "R2":        ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }


# ============================================================
# Reviewer-analysis utility functions
# ============================================================

def unseen_fraction(formula_str, unseen_elements=("Mo", "Ta", "V", "W")):
    """
    Return the summed mole fraction of 'unseen' elements (Mo, Ta, V, W)
    in a formula string such as 'Al16Mo16Nb16Ta16Ti16V16W16Zr16'.

    Used in Reviewer 2 Comment 6 / Comment 3 to bin compositions by how
    much of the 7-component space falls outside the 4-component training set.
    """
    matches = re.findall(r"([A-Z][a-z]*)(\d+)", str(formula_str))
    counts  = {elem: int(n) for elem, n in matches}
    total   = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(counts.get(el, 0) for el in unseen_elements) / total


def nmae_by_quartile(y_true_sub, y_pred_sub, q_assign_arr):
    """
    Compute NMAE separately for each of the four f_unseen quartiles.

    Parameters
    ----------
    y_true_sub   : array-like — ground-truth values for the idx_has_unseen subset.
    y_pred_sub   : array-like — model predictions for the same subset.
    q_assign_arr : int array — quartile assignment (0–3) for each sample in the subset.

    Returns
    -------
    dict  {0: nmae_Q1, 1: nmae_Q2, 2: nmae_Q3, 3: nmae_Q4}
    nan is stored when a quartile has no samples.
    """
    y_true_sub = np.asarray(y_true_sub)
    y_pred_sub = np.asarray(y_pred_sub)
    results = {}
    for q in range(4):
        mask = (q_assign_arr == q)
        if mask.sum() == 0:
            results[q] = np.nan
            continue
        results[q] = ErrorMetrics.nmae(y_true_sub[mask], y_pred_sub[mask])
    return results


def dominant_elements(formula_str, elem_set, threshold=0.5):
    """
    Return True if the combined mole fraction of elements in elem_set
    exceeds threshold for the given formula string.

    Requires pymatgen.core.Composition (imported locally to avoid a hard
    dependency when the function is not used).

    Used in the D4 internal cross-composition ablation (Comment 1):
      - elem_set = {"Al", "Nb"}  → Al+Nb-dominant compositions (training)
      - elem_set = {"Ti", "Zr"}  → Ti+Zr-dominant compositions (test)
    """
    try:
        from pymatgen.core import Composition as _Comp
        comp  = _Comp(str(formula_str).strip())
        total = sum(comp.values())
        if total == 0:
            return False
        return sum(comp.get(el, 0) for el in elem_set) / total >= threshold
    except Exception:
        return False


# ============================================================
# GPyTorch Gaussian Process — ARD-SE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    """Factory: GP with ARD squared-exponential (SE) kernel."""
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module  = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n_pcs))

        def forward(self, x):
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return ARDRBFModel(train_x, train_y, likelihood)


class GPyTorchGPR_ARDSE:
    """
    Gaussian Process Regression with ARD-SE kernel via GPyTorch.

    Input X is assumed to be already globally scaled (StandardScaler fitted
    on the full dataset). Only y is scaled locally per training batch.
    """

    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs      = n_pcs
        self.alpha      = alpha
        self.device     = DEVICE
        self.model      = None
        self.likelihood = None
        self._fitted    = False
        self.ardse      = ardse
        self.scaler_y   = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        """Fit via marginal-likelihood (MLL) optimisation."""
        Xs = np.asarray(X)
        y  = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model      = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)
        self.model      = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = -mll(self.model(X_train), y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        """Predict in original (un-scaled) units with uncertainty."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs     = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds    = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s  = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std  = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean


# ============================================================
# Active learning — Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    """
    Sequential active-learning loop using GP posterior uncertainty as the
    acquisition function.

    train_indices, X_train, and y_train are exposed as instance attributes
    so that external code (reviewer ablations) can reset the model state
    without re-instantiating the class.  _fit_model() is therefore also
    intentionally public.
    """

    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs         = n_pcs
        self.kernel        = kernel
        self.model         = None
        # Persistent training state — can be overwritten externally for ablations
        self.train_indices = None
        self.X_train       = None
        self.y_train       = None
        # History now includes NMAE to match reviewer-response reporting
        self.history = {
            "n_samples": [],
            "MAE":       [],
            "MAPE":      [],
            "NMAE":      [],
            "RMSE":      [],
            "R2":        [],
            "MeanSigma": [],
        }

    def _fit_model(self):
        """(Re-)fit GPR on the current X_train / y_train."""
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        """Return (sigma, mu, sigma) for use in acquisition scoring."""
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4,
            batch_size=1, max_samples=35, mape_threshold=2.0, random_state=42):

        X_all    = np.asarray(X_all)
        y_all    = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # Initialisation
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array(
                [any(f.strip() == s for s in elemental_symbols) for f in formulas]
            )
            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: elemental seed not found — using random initialisation.")
                np.random.seed(random_state)
                train_indices = np.random.choice(
                    len(X_all), size=min(initial_n, len(X_all)), replace=False
                )

        self.X_train       = X_all[train_indices].copy()
        self.y_train       = y_all[train_indices].copy()
        self.train_indices = train_indices.copy()
        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_rem, y_std_rem = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_rem)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_rem)))

            print(
                f"[ACTIVE] iter={iteration:02d}  n_train={len(self.X_train):3d}  "
                f"n_test={len(remaining_indices):4d}  "
                f"MAPE={metrics['MAPE']:.2f}%  NMAE={metrics['NMAE']:.2f}%  "
                f"R2={metrics['R2']:.4f}"
            )

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_rem, _, _ = self._information_gain(X_remaining)
            sel_local    = np.argsort(ig_rem)[-batch_size:]
            sel_global   = remaining_indices[sel_local]

            self.X_train       = np.vstack([self.X_train, X_all[sel_global]])
            self.y_train       = np.concatenate([self.y_train, y_all[sel_global]])
            self.train_indices = np.concatenate([self.train_indices, sel_global])
            iteration += 1

        self._fit_model()
        return self.model, self.history


# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    """Passive random-selection baseline for comparison with BED."""

    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs  = n_pcs
        self.kernel = kernel
        self.model  = None
        # History tracks NMAE alongside other metrics for fair comparison
        self.history = {
            "n_samples": [],
            "MAE":       [],
            "MAPE":      [],
            "NMAE":      [],
            "RMSE":      [],
            "R2":        [],
            "MeanSigma": [],
        }

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4,
            batch_size=1, max_samples=35, random_state=123):

        X_all    = np.asarray(X_all)
        y_all    = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array(
                [any(f.strip() == s for s in elemental_symbols) for f in formulas]
            )
            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                np.random.seed(random_state)
                train_indices = np.random.choice(
                    len(X_all), size=min(initial_n, len(X_all)), replace=False
                )
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_rem, y_std_rem = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_rem)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_rem)))

            print(
                f"[RANDOM] iter={iteration:02d}  n_train={len(X_train):3d}  "
                f"n_test={len(remaining_indices):4d}  "
                f"MAPE={metrics['MAPE']:.2f}%  NMAE={metrics['NMAE']:.2f}%  "
                f"R2={metrics['R2']:.4f}"
            )

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            sel_local  = np.random.choice(len(remaining_indices), size=k, replace=False)
            sel_global = remaining_indices[sel_local]

            X_train       = np.vstack([X_train, X_all[sel_global]])
            y_train       = np.concatenate([y_train, y_all[sel_global]])
            train_indices = np.concatenate([train_indices, sel_global])
            iteration += 1

        self.model = gpr
        return self.model, self.history


# ============================================================
# Component-count helpers
# ============================================================

def count_components(formula_str):
    """Count the number of unique elements in a formula string."""
    elements = re.findall(r"([A-Z][a-z]*)", str(formula_str))
    return len(set(elements))


def filter_by_components(X, y, formulas, min_comp=1, max_comp=4):
    """Return the subset of data whose component count is in [min_comp, max_comp]."""
    component_counts = np.array([count_components(f) for f in formulas])
    mask = (component_counts >= min_comp) & (component_counts <= max_comp)
    return X[mask], y[mask], formulas[mask], mask, component_counts


# ============================================================
# Generalisation analysis
# ============================================================

def run_generalization_zero_shot(X_full, y_full, formulas, n_pcs, tag="zero-shot"):
    """
    Train on 1–4 component compositions (up to max_samples); predict zero-shot
    on 5-, 6-, and 7-component alloys.
    Returns the fitted BED object and the global training indices.
    """
    print(f"GENERALISATION ({tag}) — train 1–4 comp, zero-shot predict 5–7 comp")

    X_tr, y_tr, f_tr, _, _ = filter_by_components(
        X_full, y_full, formulas, min_comp=1, max_comp=4
    )

    bed = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
    bed.run(X_tr, y_tr, f_tr, initial_n=10, batch_size=1, max_samples=200, random_state=42)
    return bed, bed.train_indices


def run_generalization_analysis(X, y, formulas, n_pcs, ErrorMetrics, BayesianExperimentDesign):
    """
    Train on 1–4 components, evaluate zero-shot on 5-, 6-, 7-component alloys.
    Saves parity plots and a CSV summary.
    """
    print("\n" + "=" * 80)
    print("GENERALISATION: train on 1–4 comp, predict 5–7 comp")
    print("=" * 80)

    X_tr4, y_tr4, f_tr4, _, _ = filter_by_components(X, y, formulas, 1, 4)
    X_t5,  y_t5,  _,     _, _ = filter_by_components(X, y, formulas, 5, 5)
    X_t6,  y_t6,  _,     _, _ = filter_by_components(X, y, formulas, 6, 6)
    X_t7,  y_t7,  _,     _, _ = filter_by_components(X, y, formulas, 7, 7)

    print(f"  Train (1–4 comp): {len(X_tr4)}")
    print(f"  Test  5-comp:     {len(X_t5)}")
    print(f"  Test  6-comp:     {len(X_t6)}")
    print(f"  Test  7-comp:     {len(X_t7)}")

    if len(X_t5) + len(X_t6) + len(X_t7) == 0:
        print("  No 5–7 component test data found.")
        return None, None

    active_gen = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
    active_gen.run(X_tr4, y_tr4, f_tr4, initial_n=10, batch_size=1,
                   max_samples=200, random_state=42)

    results  = {}
    n_counts = []
    for n_comp, X_test, y_test in [(5, X_t5, y_t5), (6, X_t6, y_t6), (7, X_t7, y_t7)]:
        if len(X_test) == 0:
            continue
        y_pred, y_std = active_gen.model.predict(X_test, return_std=True)
        met = ErrorMetrics.compute_all(y_test, y_pred)
        results[n_comp] = {"y_true": y_test, "y_pred": y_pred,
                           "y_std": y_std, "metrics": met}
        print(f"  {n_comp}-comp (n={len(X_test)}):  "
              f"R²={met['R2']:.4f}  MAPE={met['MAPE']:.2f}%  "
              f"NMAE={met['NMAE']:.2f}%  MAE={met['MAE']:.4f}")
        n_counts.append(len(X_test))

    if results:
        colors = ["purple", "darkorange", "teal"]
        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]
        for idx, (nc, ax) in enumerate(zip(sorted(results), axes)):
            r = results[nc]
            ax.errorbar(r["y_true"], r["y_pred"], yerr=r["y_std"],
                        fmt="o", alpha=0.6, capsize=5, c=colors[idx])
            lim = [r["y_true"].min(), r["y_true"].max()]
            ax.plot(lim, lim, "r--", lw=2)
            ax.set_xlabel("DFT Bulk Modulus (GPa)", fontweight="bold")
            ax.set_ylabel("ML Predicted (GPa)", fontweight="bold")
            ax.set_title(f"{nc}-comp  n={n_counts[idx]}", fontweight="bold")
            ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("generalization_5_7_components.png", dpi=300)
        plt.show()
        plt.close()
        print("Saved generalization_5_7_components.png")

        data = [{
            "Components": nc,
            "N_Test":     len(results[nc]["y_true"]),
            "R2":         f"{results[nc]['metrics']['R2']:.4f}",
            "MAPE_%":     f"{results[nc]['metrics']['MAPE']:.2f}",
            "NMAE_%":     f"{results[nc]['metrics']['NMAE']:.2f}",
            "MAE":        f"{results[nc]['metrics']['MAE']:.4f}",
        } for nc in sorted(results)]
        pd.DataFrame(data).to_csv("generalization_summary.csv", index=False)
        print("Saved generalization_summary.csv")

    return results, active_gen


# ============================================================
# Component-optimisation case study
# ============================================================

def run_optimization_study(
    comp_target, active_learner, X_full, y_full, formulas,
    initial_train_idx, iterations=20, add_per_iter=1, color="purple"
):
    """
    Starting from initial_train_idx (post-Phase-1 state), actively add
    samples from the global pool and evaluate per-iteration NMAE on the
    comp_target-component subset.
    """
    print(f"\n{'='*80}")
    print(f"CASE STUDY: {comp_target}-component optimisation")
    print(f"{'='*80}")

    active_learner.train_indices = initial_train_idx.copy()
    active_learner.X_train       = X_full[initial_train_idx].copy()
    active_learner.y_train       = y_full[initial_train_idx].copy()
    active_learner._fit_model()

    comp_counts = np.array([count_components(f) for f in formulas])
    idx_target  = np.where(comp_counts == comp_target)[0]
    if len(idx_target) == 0:
        print(f"  No {comp_target}-component samples found.")
        return

    print(f"  {len(idx_target)} {comp_target}-component samples for validation.")

    hist = {"n_added": [], "MAE": [], "MAPE": [], "NMAE": [], "RMSE": [], "R2": []}
    all_indices = np.arange(len(X_full))

    for i in range(iterations):
        y_pred, _ = active_learner.model.predict(X_full[idx_target], return_std=True)
        met = ErrorMetrics.compute_all(y_full[idx_target], y_pred)
        hist["n_added"].append(i)
        hist["MAE"].append(met["MAE"])
        hist["MAPE"].append(met["MAPE"])
        hist["NMAE"].append(met["NMAE"])
        hist["RMSE"].append(met["RMSE"])
        hist["R2"].append(met["R2"])
        print(f"  [iter {i+1:2d}] {comp_target}-comp  "
              f"MAE={met['MAE']:.4f}  MAPE={met['MAPE']:.2f}%  "
              f"NMAE={met['NMAE']:.2f}%  R²={met['R2']:.4f}")

        remaining = np.setdiff1d(all_indices, active_learner.train_indices)
        if len(remaining) == 0:
            print("  Pool exhausted.")
            break

        sigma_rem, _, _ = active_learner._information_gain(X_full[remaining])
        sel_local  = np.argsort(sigma_rem)[-add_per_iter:]
        sel_global = remaining[sel_local]

        active_learner.X_train = np.vstack([active_learner.X_train, X_full[sel_global]])
        active_learner.y_train = np.concatenate([active_learner.y_train, y_full[sel_global]])
        active_learner.train_indices = np.concatenate(
            [active_learner.train_indices, sel_global]
        )
        active_learner._fit_model()

    y_pred_fin, y_std_fin = active_learner.model.predict(X_full[idx_target], return_std=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{comp_target}-Component Optimisation", fontweight="bold")

    ax[0].plot(hist["n_added"], hist["MAE"], "o-", color=color)
    ax[0].set_xlabel("Additional samples added")
    ax[0].set_ylabel(f"MAE on {comp_target}-comp subset")
    ax[0].set_title("Error reduction")
    ax[0].grid(True, alpha=0.3)

    ax[1].errorbar(y_full[idx_target], y_pred_fin, yerr=y_std_fin, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor=color, markeredgecolor="k")
    ax[1].plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], "k--")
    ax[1].set_xlabel("Ground truth (GPa)")
    ax[1].set_ylabel("Prediction (GPa)")
    ax[1].set_title(f"Final {comp_target}-comp  R²={hist['R2'][-1]:.3f}")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f"Extrapolate{comp_target}comp.png", dpi=300)
    plt.show()
    plt.close()


# ============================================================
# Plotting helpers
# ============================================================

def process_alloy_data(df, output_csv_path=None):
    """Parse formula strings to element-count and percentage columns."""
    def parse_formula(formula):
        matches = re.findall(r"([A-Z][a-z]*)(\d+)", str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed     = df["formula"].apply(parse_formula)
    df_counts  = pd.DataFrame(parsed.tolist()).fillna(0).astype(int)
    for elem in ["Al", "Nb", "Ti", "Zr"]:
        if elem not in df_counts.columns:
            df_counts[elem] = 0
    df_counts = df_counts[["Al", "Nb", "Ti", "Zr"]]
    df_final  = pd.concat([df, df_counts], axis=1)
    for elem in ["Al", "Nb", "Ti", "Zr"]:
        df_final[f"{elem}_pct"] = (df_final[elem] / 128) * 100
    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final


def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum  = np.cumsum(vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, n_show + 1)
    ax.bar(idx, vals, color="#2E86AB", alpha=0.7, label="Individual")
    ax.plot(idx, cum, "o-r", linewidth=2, markersize=6, label="Cumulative")
    ax.set_xlabel("Principal Component", fontweight="bold")
    ax.set_ylabel("Variance Explained (%)", fontweight="bold")
    ax.set_xticks(idx)
    ax.grid(True, alpha=0.3)
    ax.set_title("PCA Variance Explained", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()


def plot_convergence(active_hist, random_hist, property_name,
                     fname="Figure_6_Convergence.png"):
    """Plot NMAE, R², MAE, RMSE convergence curves for active vs. random."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence — {property_name}", fontweight="bold")

    # NMAE panel
    ax = axes[0, 0]
    ax.axhline(y=2, color="grey", linestyle="--", linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["NMAE"],
            "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["NMAE"],
            "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("NMAE (%)", fontweight="bold")
    ax.set_title("NMAE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # R² panel
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"],
            "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"],
            "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_title("R²", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # MAE panel
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"],
            "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"],
            "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("MAE (GPa)", fontweight="bold")
    ax.set_title("MAE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # RMSE panel
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"],
            "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"],
            "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("RMSE (GPa)", fontweight="bold")
    ax.set_title("RMSE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {fname}")


def plot_predictions(y_true, y_pred, y_std, property_name,
                     fname="Figure_7_Predictions.png", color=None):
    """Parity plot + residual panel with ±2σ band."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions — {property_name}", fontweight="bold")

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    m    = (vmax - vmin) * 0.05

    ax = axes[0]
    ax.plot([vmin - m, vmax + m], [vmin - m, vmax + m], "k--", linewidth=2)
    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray",
                capsize=3, alpha=0.7, markersize=5,
                markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth (GPa)", fontweight="bold")
    ax.set_ylabel("Prediction (GPa)", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8,
               edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction (GPa)", fontweight="bold")
    ax.set_ylabel("Residual (GPa)", fontweight="bold")
    ax.set_title("Residuals", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {fname}")


def plot_uncertainty_hist(y_true, y_pred, y_std, property_name,
                          fname="Figure_7b_Uncertainty.png", color=None):
    """Histogram of predicted σ with coverage statistics."""
    errors   = np.abs(y_true - y_pred)
    within_1 = np.mean(errors <= y_std) * 100.0
    within_2 = np.mean(errors <= 2.0 * y_std) * 100.0

    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--",
               label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty — {property_name}", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    text = (f"Within ±1σ: {within_1:.1f}% (theory 68%)\n"
            f"Within ±2σ: {within_2:.1f}% (theory 95%)")
    ax.text(0.98, 0.98, text, transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=8)
    ax.legend(loc="lower right")
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {fname}")