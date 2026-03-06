"""
Auto-generated module from Jupyter Notebook
"""

import sys
import numpy as np
import math
import h5py
import os
import re
import matplotlib.pyplot as plt

DIR='/content/drive'
HOME = os.path.join(DIR, 'MyDrive')
HOMED = os.path.join(HOME, "Analyze7RHEA")
os.chdir(HOMED)

import os
import warnings
warnings.filterwarnings("ignore")

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import warnings
warnings.filterwarnings("ignore")

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Use a small epsilon relative to the max value to stabilize
        # the denominator, preventing explosion when y_true is near zero.
        epsilon = 1e-6 * np.max(np.abs(y_true))

        # Calculate percentage error, stabilizing the denominator
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))

        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true)#np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    """Factory function to create ARDRBFModel with correct n_pcs closure."""
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            #self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])#ConstantMean()
            #self.mean_module = torch.tenso(1e-2) #ConstantMean()
            self.mean_module = ConstantMean()
            #self.covar_module = ScaleKernel(gpytorch.kernels.LinearKernel())
            #self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n_pcs))
            #self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n_pcs)+gpytorch.kernels.PolynomialKernel(power=2))
            #self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=n_pcs))
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1) #works best for now
            #self.covar_module = LCMKernel(base_kernels=[gpytorch.kernels.LinearKernel(),RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)
            #self.covar_module = ScaleKernel(gpytorch.kernels.LinearKernel()+RBFKernel(ard_num_dims=n_pcs))
            #self.covar_module = LCMKernel(base_kernels=[gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=n_pcs)], num_tasks=1)
            #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=3))
            """self.covar_module = gpytorch.kernels.ScaleKernel(
                                                                  # power=2 allows for quadratic trends (parabolic shapes)
                                                                  gpytorch.kernels.PolynomialKernel(power=2) +
                                                                  # ARD RBF for the local residuals
                                                                  gpytorch.kernels.RBFKernel(ard_num_dims=n_pcs)
                                                              )"""


        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

def create_ard_rbf_model2(train_x, train_y, likelihood, n_pcs):
    """Factory function to create ARDRBFModel with correct n_pcs closure."""
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            #self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n_pcs))
            #self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=n_pcs))
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1) #works best for now
            #self.covar_module = LCMKernel(base_kernels=[gpytorch.kernels.LinearKernel(),RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)
            #self.covar_module = ScaleKernel(gpytorch.kernels.LinearKernel()+RBFKernel(ard_num_dims=n_pcs))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)


class GPyTorchGPR_ARDSE:
    """
    Gaussian Process Regression with ARDSE kernel using GPyTorch.
    NOTE: It expects input X to be ALREADY globally scaled PCA features.
    """

    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse

        # Only Y is scaled locally. X is assumed globally scaled.
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        """Fit GPyTorch model with marginal likelihood optimization."""
        # X (PCA features) is ALREADY globally scaled
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # Fit scaler on THIS training set y (local scaling)
        ys = self.scaler_y.fit_transform(y).ravel()

        # Convert to torch tensors
        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        # Build model using factory function
        self.likelihood = GaussianLikelihood()
        if self.ardse=='ardse':
            self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)
        elif self.ardse=='ardse2':
            self.model = create_ard_rbf_model2(X_train, y_train, self.likelihood, self.n_pcs)

        # Move to device and train
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        """Predict with uncertainty quantification, returning ORIGINAL units."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)
        Xs = X # X is ALREADY globally scaled

        # Convert to torch tensor
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        # Inverse transform mean to get original units
        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()

        # Rescale standard deviation to original units
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }

    def _fit_model(self):
        """Fit GPyTorch model on training set."""
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        """
        Compute information gain: absolute uncertainty σ(x).
        """
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION STRATEGY ---
        if initial_indices is not None:
            # Use externally provided indices for Apple-to-Apples comparison
            train_indices = np.array(initial_indices)
            self.X_train = X_all[train_indices].copy()
            self.y_train = y_all[train_indices].copy()
        else:
            # Internal Initialization logic (Fall back)
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
                self.X_train = X_all[train_indices].copy()
                self.y_train = y_all[train_indices].copy()
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
                self.X_train = X_all[train_indices].copy()
                self.y_train = y_all[train_indices].copy()

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            # Fit model
            self._fit_model()

            # Get remaining
            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            # Evaluate
            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(
                f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                f"n_test={len(remaining_indices):4d}, "
                f"MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}"
            )

            # Convergence check
            if metrics["MAPE"] <= mape_threshold:
                print(f"  → Converged (MAPE ≤ {mape_threshold:.2f} %)")
                #break

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            # Compute information gain and select
            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            # Add selected samples
            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION STRATEGY ---
        if initial_indices is not None:
            # Use externally provided indices for Apple-to-Apples comparison
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            # Internal Initialization logic (Fall back)
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
                X_train = X_all[train_indices].copy()
                y_train = y_all[train_indices].copy()
            else:
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
                X_train = X_all[train_indices].copy()
                y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            # Fit model
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            # Get remaining indices
            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            # Evaluate
            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(
                f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                f"n_test={len(remaining_indices):4d}, "
                f"MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}"
            )

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            # Randomly select from REMAINING samples
            k = min(batch_size, len(remaining_indices))

            # Need to re-seed here to ensure random walk is different for different calls if needed,
            # though usually controlled by loop external seed.
            np.random.seed(random_state + iteration)

            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history


# ============================================================
# Plotting helpers (Unchanged)
# ============================================================

def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(1, n_show + 1)
    ax.bar(idx, vals, color="#2E86AB", alpha=0.7, label="Individual")
    ax.plot(idx, cum, "o-r", linewidth=2, markersize=6, label="Cumulative")
    ax.set_xlabel("Principal Component", fontweight="bold")
    ax.set_ylabel("Variance Explained (%)", fontweight="bold")
    ax.set_xticks(idx)
    ax.grid(True, alpha=0.3)
    ax.set_title("PCA Variance Explained (Analogue of Figure 3)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", target_col2="formation_energy",
                          fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3:
        print("Need at least 3 PC columns for Figure 5.")
        return

    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values
    #target2 = df[target_col2].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),#^
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),#s
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o') #D
    ]

    fig = plt.figure(figsize=(14, 5))

    # Bulk modulus plot
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis",
                      s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow",
                      s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)


    # --- 4. Plot Each Layer ---
    c=0
    for label, mask, color, marker in layers:
        subset = df[mask]

        if not subset.empty:
            if n==98:
                if c==0:
                  c+=1
                  #ax2.scatter(subset['PC1'], subset['PC2'], linewidths=2, edgecolors='black', label = 'Pure element', s=100, marker=marker, facecolors='None')
                #else:
                  #ax2.scatter(subset['PC1'], subset['PC2'], linewidths=2, edgecolors='black', s=100, marker=marker, facecolors='None')
                #ax2.legend()

    """for label, mask, color, marker in layers:
        subset = df[mask]

        if not subset.empty:
            if n==98:
                if c==0:
                  c+=1
                  ax1.scatter(subset['PC2'], subset['PC3'], linewidths=2, edgecolors='black', label = 'Pure element', s=100, marker=marker, facecolors='None')
                else:
                  ax1.scatter(subset['PC2'], subset['PC3'], linewidths=2, edgecolors='black', s=100, marker=marker, facecolors='None')
                #ax1.legend()"""

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    #plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

def plot_convergence(active_hist, random_hist, property_name,
                     fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)#, label='')
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-",
            label="Random Selection", color="maroon", linewidth=2) ##A23B72
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-",
            label="Active Selection", color="limegreen", linewidth=2) ##2E86AB
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("MAPE (%)", fontweight="bold")
    ax.set_title("MAPE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-",
            label="Random Selection", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-",
            label="Active Selection", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_title("R²", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-",
            label="Random Selection", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-",
            label="Active Selection", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("MAE", fontweight="bold")
    ax.set_title("MAE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-",
            label="Random Selection", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-",
            label="Active Selection", color="limegreen", linewidth=2)
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("RMSE", fontweight="bold")
    ax.set_title("RMSE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

def plot_predictions(y_true, y_pred, y_std, property_name,
                     fname="Figure_7_Predictions.png", color = None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    # Left: y_true vs y_pred with error bars
    ax = axes[0]

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],'k--', linewidth=2)

    if color ==None:
        ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray",
                    capsize=3, alpha=0.7, markersize=5,
                    markeredgecolor="k", markerfacecolor="#2E86AB")
    else:
        ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray",
                    capsize=3, alpha=0.7, markersize=5,
                    markeredgecolor="k", markerfacecolor=color)
    ymin, ymax = np.min(y_true), np.max(y_true)
    #ax.plot([ymin, ymax], [ymin, ymax], "r--")
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    #ax.set_title("Predictions with uncertainty", fontweight="bold")
    #ax.grid(True, alpha=0.3)
    #ax.legend()
    ax.set_box_aspect(1)

    # Right: residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k",
               facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual (truth - pred)", fontweight="bold")
    ax.set_title("Residuals", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name,
                          fname="Figure_7b_Uncertainty.png", color = None):
    errors = np.abs(y_true - y_pred)
    within_1 = np.mean(errors <= y_std) * 100.0
    within_2 = np.mean(errors <= 2.0 * y_std) * 100.0

    fig, ax = plt.subplots(figsize=(6, 4))
    if color == None:
        ax.hist(y_std, bins=30, color="#2E86AB", alpha=0.8, edgecolor="k")
    else:
        ax.hist(y_std, bins=30, color=color, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--",
               label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_box_aspect(1)

    text = (f"Coverage:\n"
            f"Within ±1σ: {within_1:.1f}% (theoretical 68%)\n"
            f"Within ±2σ: {within_2:.1f}% (theoretical 95%)")
    ax.text(0.98, 0.98, text, transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=8)
    ax.legend(loc='lower right')
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

    fig, ax = plt.subplots(figsize=(6, 4))
    if color == None:
        ax.hist(y_std, bins=30, color="#2E86AB", alpha=0.8, edgecolor="k")
    else:
        ax.hist(y_std, bins=30, color=color, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--",
               label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_box_aspect(1)

    text = (f"Coverage:\n"
            f"Within ±1σ: {within_1:.1f}% (theoretical 68%)\n"
            f"Within ±2σ: {within_2:.1f}% (theoretical 95%)")
    """ax.text(0.98, 0.98, text, transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=8)
    ax.legend(loc='lower right')"""
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f"opu_{fname}", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

import pandas as pd
import re

def process_alloy_data(df, output_csv_path=None):
    # 1. Load the dataset
    #df = pd.read_csv(input_csv_path)

    # 2. Define a parser function
    def parse_formula(formula):
        """
        Parses a string like 'Al8Nb92Ti20Zr8' into a dictionary {'Al': 8, 'Nb': 92, ...}
        """
        # Regex explanation:
        # ([A-Z][a-z]*) : Matches the element symbol (e.g., "Al", "Nb")
        # (\d+)         : Matches the integer count immediately following it
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    # 3. Apply the parser to the formula column
    # This creates a list of dictionaries, one for each row
    parsed_counts = df['formula'].apply(parse_formula)

    # 4. Convert the parsed data into a DataFrame and merge
    # This automatically creates columns for every element found (Al, Nb, Ti, Zr)
    df_counts = pd.DataFrame(parsed_counts.tolist())

    # Fill missing elements with 0 (e.g., if a formula is just Nb128, Al becomes 0)
    df_counts = df_counts.fillna(0).astype(int)

    # Optional: Enforce specific column order or specific elements
    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    # Ensure all target columns exist (adds them as 0 if they were never found in any row)
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    # Filter to keep only the target elements in the desired order
    df_counts = df_counts[target_elements]

    # Combine original data with the new count columns
    df_final = pd.concat([df, df_counts], axis=1)

    # 5. Add Percentage Columns
    # Since total atoms = 128, Percentage = (Count / 128) * 100
    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    # 6. Save or Display
    if output_csv_path!=None:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

def pcquad(df, pc_cols, n, target_col="bulk_modulus", target_col2="formation_energy",
                          fname="pcquad.png"):
    if len(pc_cols) < 3:
        print("Need at least 3 PC columns for Figure 5.")
        return

    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values
    #target2 = df[target_col2].values

    fig = plt.figure(figsize=(14,14))

    # Bulk modulus plot
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis",
                      s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow",
                      s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'],
            c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)
    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),#^
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),#s
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o') #D
    ]

    c=0
    # --- 4. Plot Each Layer ---
    for label, mask, color, marker in layers:
        subset = df[mask]

        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'],
                        c=color,          # Color defined above
                        label=label,      # Legend label
                        marker=marker,    # Different markers help distinguish overlaps
                        s=100,             # Slightly larger size for highlights
                        alpha=0.8,        # Slight transparency
                        edgecolors='k',   # Black edge to make points pop
                        linewidth=0.5,
                        zorder=2)         # Ensure these sit on top of the gray points

            if n==98:
                if c==0:
                  c+=1
                  #ax2.scatter(subset['PC1'], subset['PC2'], edgecolors='black', label = 'Pure element', s=100, marker=marker, facecolors='None')
                #else:
                  #ax2.scatter(subset['PC1'], subset['PC2'], edgecolors='black', s=100, marker=marker, facecolors='None')
                #ax2.legend()

    # --- 5. Final Formatting ---
    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")
    #ax0.set_title(f'PCA Map: Elemental High-Concentration Zones (n={n}%)')
    #ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
    #ax0.legend()
    #ax0.grid(True, linestyle='--', alpha=0.3)
    #ax0.tight_layout()

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'],
            c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    # --- 4. Plot Each Layer ---
    for label, mask, color, marker in layers:
        subset = df[mask]

        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'],
                        c=color,          # Color defined above
                        label=label,      # Legend label
                        marker=marker,    # Different markers help distinguish overlaps
                        s=100,             # Slightly larger size for highlights
                        alpha=0.8,        # Slight transparency
                        edgecolors='k',   # Black edge to make points pop
                        linewidth=0.5,
                        zorder=2)         # Ensure these sit on top of the gray points

    # --- 5. Final Formatting ---
    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")
    #ax0.set_title(f'PCA Map: Elemental High-Concentration Zones (n={n}%)')
    #ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
    #ax0.legend()
    #ax0.grid(True, linestyle='--', alpha=0.3)
    #ax0.tight_layout()

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    #plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

# ============================================================
# Main script - GLOBAL X SCALING, single-property (bulk_modulus)
# ============================================================

df = pd.read_csv('pca_pspall7.csv') #pca_pspall7 #pca_psp7
df4 = pd.read_csv('psp4all.csv')

# If this is a small stub file, try to find the full one
if len(df) < 1000:
    print("Looking for full dataset with 4495 samples...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        print(f"Found full dataset: {csv_path} ({os.path.getsize(csv_path)} bytes)")
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# ---- REQUIRED COLUMNS (single target) ----
required_cols = ["bulk_modulus"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    if df[col].isna().any():
        raise ValueError(f"Column {col} has NaN values. Please provide complete dataset.")

# PC columns
pc_cols = [c for c in df.columns if c.startswith("PC")]
if len(pc_cols) < 3:
    raise ValueError("Need at least PC1, PC2, PC3 in the CSV.")

print(df.head)

# Use first 3 PCs as features
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values          # unscaled PCA features
#selected_cols = pc_cols[:2] + [pc_cols[6]]
#selected_cols = pc_cols[:2] + pc_cols[6]
#X_raw = df[selected_cols].values
#X_raw = df.iloc[:, [0, 1, 5, 6]].values
y_bm  = df["bulk_modulus"].values          # ONLY property

# ------------------------------------------------------------
# GLOBAL normalization of PCA inputs (X)
# ------------------------------------------------------------
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formula column for initialization (works for both 4‑ and 7‑component)
formula_col = None
for col in ["formula", "composition", "comp", "Formula"]:
    if col in df.columns:
        formula_col = col
        break

if formula_col is None:
    print("Warning: No formula column found. Will use synthetic labels.")
    formulas = np.array([f"Sample{i}" for i in range(len(df))])
else:
    formulas = df[formula_col].values

print(f"✓ Loaded dataset: {len(df)} samples (4‑ and 7‑component combined)")
print(f"Feature matrix shape: {X.shape}")
print(f"Bulk modulus range: {y_bm.min():.2f} – {y_bm.max():.2f} GPa")

# --- PCA variance explained (unchanged, just uses all data) ---
plot_figure3_pca_variance(df, pc_cols)

# --- Elemental parsing and PC maps for the 4‑component Al–Nb–Ti–Zr subset ---
# This will naturally only highlight rows that actually contain those elements;
# 7‑component NbMoTiTaVWZr rows will just appear as "all data" background.
df_final = process_alloy_data(df)
df4 = process_alloy_data(df4)
n = 75  # e.g., 75% threshold

if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# Simple 2D PC1–PC2 map with high‑concentration overlays
plt.figure(figsize=(10, 8))
plt.scatter(df_final['PC1'], df_final['PC2'],
            c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

layers = [
    (f'Al > {n}%', df_final['Al_pct'] > n, 'red',    'o'),
    (f'Ti > {n}%', df_final['Ti_pct'] > n, 'blue',   'o'),
    (f'Zr > {n}%', df_final['Zr_pct'] > n, 'green',  'o'),
    (f'Nb > {n}%', df_final['Nb_pct'] > n, 'orange', 'o'),
]

for label, mask, color, marker in layers:
    subset = df_final[mask]
    if not subset.empty:
        plt.scatter(subset['PC1'], subset['PC2'],
                    c=color,
                    label=label,
                    marker=marker,
                    s=100,
                    alpha=0.8,
                    edgecolors='k',
                    linewidth=0.5,
                    zorder=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA Map: Elemental High-Concentration Zones (n={n}%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# UNIFIED INITIALIZATION STRATEGY (still 4 pure elements)
# Train up to 45 samples (mostly 4‑component), predict on ALL
# (4‑ and 7‑component) in X.
# ------------------------------------------------------------
initial_n   = 10
max_samples = 200

elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols)
                           for f in formulas])

if np.sum(elemental_mask) >= 4:
    initial_train_indices = np.where(elemental_mask)[0][:4]
    print(f"\n[INIT] Using Elemental Initialization: {initial_train_indices}")
    print(f"       Formulas: {formulas[initial_train_indices]}")
else:
    np.random.seed(42)
    initial_train_indices = np.random.choice(len(df4['formula'].values), size=initial_n, replace=False)
    print(f"\n[INIT] Elemental samples not found. Using Fixed Random Initialization: {initial_train_indices}")

print("\n" + "="*80)
print("BULK MODULUS - ACTIVE vs RANDOM (single‑property GP)")
print(f"Training on {initial_n}→{max_samples} samples, predicting on ALL remaining (4‑ and 7‑component).")
print("NOTE: PCA inputs are globally scaled.")
print("="*80)
formulas=df4['formula'].values

# ------------------- ACTIVE SELECTION ------------------------
active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
model_bm_active, hist_bm_active = active_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    mape_threshold=2.0,
    random_state=42,
)

# ------------------- RANDOM SELECTION ------------------------
random_bm = RandomSamplingBaseline(n_pcs=n_pcs, kernel="ardse")
model_bm_rand, hist_bm_rand = random_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    random_state=123,
)

# Final predictions on FULL dataset (4‑ and 7‑component)
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)

metrics_bm = ErrorMetrics.compute_all(y_bm, y_bm_pred)
print("\nFinal Bulk Modulus metrics (Active selection, original units):")
for k, v in metrics_bm.items():
    if k in ("MAPE", "NMAE"):
        print(f"  {k:10s}: {v:8.3f} %")
    else:
        print(f"  {k:10s}: {v:8.4f}")

# Convergence + prediction plots (bulk modulus only)
plot_convergence(
    hist_bm_active, hist_bm_rand,
    property_name="Bulk Modulus",
    fname="Figure_6_Convergence_BulkModulus.png",
)
plot_predictions(
    y_bm, y_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Figure_7_Predictions_BulkModulus.png",
    color="#8ebc8e",
)
plot_uncertainty_hist(
    y_bm, y_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Figure_7b_Uncertainty_BulkModulus.png",
    color="#8ebc8e",#28a99e
)

# Simple metrics comparison table: Active vs Random (bulk_modulus only)
rand_bm_pred, y_bm_std = random_bm.model.predict(X, return_std=True)
metrics_bm_rand = ErrorMetrics.compute_all(y_bm, rand_bm_pred)

rows = []
rows.append({
    "Property": "Bulk Modulus",
    "Strategy": "Active",
    **metrics_bm,
})
rows.append({
    "Property": "Bulk Modulus",
    "Strategy": "Random",
    **metrics_bm_rand,
})

plot_predictions(
    y_bm, rand_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Random_Predictions_BulkModulus.png",
    color="#8ebc8e",
)
plot_uncertainty_hist(
    y_bm, rand_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Random_Uncertainty_BulkModulus.png",
    color="#8ebc8e",#28a99e
)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("metrics_comparison_bulk_only.csv", index=False)
print("\nSaved metrics_comparison_bulk_only.csv:")
print(metrics_df.to_string(index=False))


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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true) #np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.train_indices = None
        self.X_train = None
        self.y_train = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, NMAE={metrics['NMAE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, NMAE={metrics['NMAE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Generalization Helper Functions
# ============================================================

def count_components(formula_str):
    """Count unique elements in formula string"""
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula_str))
    components = [elem for elem, count in matches if elem and elem.strip()]
    return len(components)

def filter_by_components(X, y, formulas, min_comp=1, max_comp=4):
    """Split data by component count"""
    component_counts = np.array([count_components(f) for f in formulas])
    mask = (component_counts >= min_comp) & (component_counts <= max_comp)
    return X[mask], y[mask], formulas[mask], mask, component_counts

def run_generalization_analysis(X, y, formulas, n_pcs, ErrorMetrics, BayesianExperimentDesign):
    """
    Train on 1-4 components → Predict on 5-7 components
    Returns: R², MAPE, MAE for each component group + plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS: Training on 1-4 Components")
    print("="*80)

    # Split data
    X_train_up4, y_train_up4, formulas_train_up4, mask_train_up4, comp_count_all = \
        filter_by_components(X, y, formulas, min_comp=1, max_comp=4)

    X_test_5, y_test_5, formulas_test_5, _, _ = \
        filter_by_components(X, y, formulas, min_comp=5, max_comp=5)

    X_test_6, y_test_6, formulas_test_6, _, _ = \
        filter_by_components(X, y, formulas, min_comp=6, max_comp=6)

    X_test_7, y_test_7, formulas_test_7, _, _ = \
        filter_by_components(X, y, formulas, min_comp=7, max_comp=7)

    # Report
    print(f"\nTraining: {len(X_train_up4)} samples (1-4 components)")
    print(f"Test 5-comp: {len(X_test_5)}")
    print(f"Test 6-comp: {len(X_test_6)}")
    print(f"Test 7-comp: {len(X_test_7)}")

    if len(X_test_5) + len(X_test_6) + len(X_test_7) == 0:
        print("\n⚠️  No test data with 5-7 components found!")
        return None, None

    # Train on 1-4
    print(f"\nTraining model on 1-4 component samples...")
    active_gen = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
    model, hist = active_gen.run(
        X_train_up4, y_train_up4, formulas_train_up4,
        initial_indices=None,
        initial_n=10,
        batch_size=1,
        max_samples=200,
        random_state=42
    )

    # Predict on 5-7
    results = {}
    test_configs = [(5, X_test_5, y_test_5), (6, X_test_6, y_test_6), (7, X_test_7, y_test_7)]
    X_test_n = []

    print(f"\nGeneralization Performance:\n")
    for comp_num, X_test, y_test in test_configs:
        if len(X_test) == 0:
            continue

        y_pred, y_std = active_gen.model.predict(X_test, return_std=True)
        metrics = ErrorMetrics.compute_all(y_test, y_pred)

        results[comp_num] = {'y_true': y_test, 'y_pred': y_pred, 'y_std': y_std, 'metrics': metrics}

        print(f"  {comp_num}-component (n={len(X_test)}):")
        print(f"    R²:    {metrics['R2']:.4f}")
        print(f"    MAPE:  {metrics['MAPE']:.2f}%")
        print(f"    NMAE:  {metrics['NMAE']:.2f}%")
        print(f"    MAE:   {metrics['MAE']:.4f}")
        print()
        X_test_n.append(len(X_test))

    # Plot
    n_plots = len(results)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        colors=['purple','darkorange','teal']

        for idx, (comp_num, ax) in enumerate(zip(sorted(results.keys()), axes)):
            r = results[comp_num]
            y_true, y_pred, y_std = r['y_true'], r['y_pred'], r['y_std']

            ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6, capsize=5, c=colors[idx])
            y_lim = [y_true.min(), y_true.max()]
            ax.plot(y_lim, y_lim, 'r--', lw=2, label='Perfect')
            ax.set_xlabel('DFT-computed Bulk Moduli (GPa)', fontweight='bold')
            ax.set_ylabel('ML-predicted Bulk Moduli (GPa)', fontweight='bold')
            ax.set_title(f'{comp_num}-comp\n n={X_test_n[idx]}', fontweight='bold')
            #ax.grid(True, alpha=0.3)
            ax.set_box_aspect(1)
            #ax.legend()

        #plt.tight_layout()
        plt.savefig('generalization_5_7_components.png', dpi=300)
        print(f"✓ Saved: generalization_5_7_components.png")
        plt.show()

    # CSV
    data = [{
        'Components': comp_num,
        'N_Test': len(results[comp_num]['y_true']),
        'R²': f"{results[comp_num]['metrics']['R2']:.4f}",
        'MAPE_%': f"{results[comp_num]['metrics']['MAPE']:.2f}",
        'NMAE_%': f"{results[comp_num]['metrics']['NMAE']:.2f}",
        'MAE': f"{results[comp_num]['metrics']['MAE']:.4f}",
    } for comp_num in sorted(results.keys())]

    df = pd.DataFrame(data)
    df.to_csv('generalization_summary.csv', index=False)
    print(f"✓ Saved: generalization_summary.csv")

    return results, active_gen

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAPE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

# ============================================================
# Helper Function for Case Studies
# ============================================================
def run_optimization_study(
    comp_target,
    active_learner,
    X_full,
    y_full,
    formulas,
    initial_train_idx,
    iterations=20,
    add_per_iter=1,
    color="purple"
):
    """
    Runs an optimization study for a specific number of components (comp_target).
    It resets the learner to the 'initial_train_idx' state (Post-Phase 1)
    before starting, ensuring studies are independent.
    """
    print(f"\n{'='*80}")
    print(f"CASE STUDY: {comp_target}-Component Optimization")
    print(f"Focus: Add points from pool to improve {comp_target}-comp samples specifically.")
    print(f"{'='*80}")

    # 1. Reset Model to End of Phase 1 State
    # (Important so studies don't contaminate each other)
    active_learner.train_indices = initial_train_idx.copy()
    active_learner.X_train = X_full[initial_train_idx].copy()
    active_learner.y_train = y_full[initial_train_idx].copy()
    active_learner._fit_model() # Refit to ensure state is correct

    # 2. Identify Target Validation Set
    comp_counts = np.array([count_components(f) for f in formulas])
    idx_target = np.where(comp_counts == comp_target)[0]

    if len(idx_target) == 0:
        print(f"No {comp_target}-component samples found!")
        return

    print(f"Found {len(idx_target)} {comp_target}-component samples for validation.")

    hist_study = {
        "n_added": [],
        "MAE": [],
        "MAPE": [],
        "NMAE": [],
        "RMSE": [],
        "R2": []
    }

    all_indices = np.arange(len(X_full))

    for i in range(iterations):
        # A. Evaluate on Target Subset ONLY
        y_pred, _ = active_learner.model.predict(X_full[idx_target], return_std=True)
        metrics = ErrorMetrics.compute_all(y_full[idx_target], y_pred)

        hist_study["n_added"].append(i)
        hist_study["MAE"].append(metrics["MAE"])
        hist_study["MAPE"].append(metrics["MAPE"])
        hist_study["NMAE"].append(metrics["NMAE"])
        hist_study["RMSE"].append(metrics["RMSE"])
        hist_study["R2"].append(metrics["R2"])

        print(f"[Iter {i+1}] {comp_target}-Comp MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, NMAE={metrics['NMAE']:.2f}%, R2={metrics['R2']:.4f}")

        # B. Select new points from the REMNANT pool
        remaining_indices = np.setdiff1d(all_indices, active_learner.train_indices)
        X_remaining = X_full[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        sigma_rem, _, _ = active_learner._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_learner.X_train = np.vstack([active_learner.X_train, X_full[select_global_idx]])
        active_learner.y_train = np.concatenate([active_learner.y_train, y_full[select_global_idx]])
        active_learner.train_indices = np.concatenate([active_learner.train_indices, select_global_idx])

        # D. Retrain
        active_learner._fit_model()

    # Final Evaluation
    y_pred_final, y_std_final = active_learner.model.predict(X_full[idx_target], return_std=True)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{comp_target}-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_study["n_added"], hist_study["MAE"], "o-", color=color)
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel(f"MAE on {comp_target}-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot
    ax[1].errorbar(y_full[idx_target], y_pred_final, yerr=y_std_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor=color, markeredgecolor="k")
    ax[1].plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final {comp_target}-Comp Prediction\n(R2={hist_study['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f"Extrapolate{comp_target}comp.png", dpi=300)
    plt.show()

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv')
df4 = pd.read_csv('psp4all.csv')

if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Generalization Analysis (Replaces standard Active Learning)
# ------------------------------------------------------------

results, active_bm = run_generalization_analysis(
    X=X,                                    # Your features
    y=y_bm,                                 # Your target (bulk modulus)
    formulas=formulas,                      # Formula strings
    n_pcs=n_pcs,                           # Number of PCs
    ErrorMetrics=ErrorMetrics,
    BayesianExperimentDesign=BayesianExperimentDesign
)

# Plot Phase 1 Results (Predictions on whole dataset using 1-4 component model)
if active_bm is not None:
    y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
    plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1: Generalization)", color="#28a99e")

    # Store the indices at the end of Phase 1 to use as the starting point for all studies
    indices_post_phase1 = active_bm.train_indices.copy()

    # ------------------------------------------------------------
    # COMPONENT OPTIMIZATION STUDIES (5, 6, 7)
    # ------------------------------------------------------------

    # Study 1: 5-Component Optimization
    run_optimization_study(
        comp_target=5,
        active_learner=active_bm,
        X_full=X,
        y_full=y_bm,
        formulas=formulas,
        initial_train_idx=indices_post_phase1,
        iterations=20,
        add_per_iter=1,
        color="purple"
    )

    # Study 2: 6-Component Optimization
    run_optimization_study(
        comp_target=6,
        active_learner=active_bm,
        X_full=X,
        y_full=y_bm,
        formulas=formulas,
        initial_train_idx=indices_post_phase1,
        iterations=20,
        add_per_iter=1,
        color="darkorange"
    )

    # Study 3: 7-Component Optimization
    run_optimization_study(
        comp_target=7,
        active_learner=active_bm,
        X_full=X,
        y_full=y_bm,
        formulas=formulas,
        initial_train_idx=indices_post_phase1,
        iterations=20,
        add_per_iter=1,
        color="teal"
    )

    print("\nAll studies completed.")
else:
    print("Skipping optimization studies due to missing generalization model.")

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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true)#np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.train_indices = None
        self.X_train = None
        self.y_train = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, NMAE={metrics['NMAE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, NMAE={metrics['NMAE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAPE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

def count_components(formula_str):
    elements = re.findall(r'([A-Z][a-z]*)', str(formula_str))
    return len(set(elements))

# ============================================================
# Helper Function for Case Studies
# ============================================================
def run_optimization_study(
    comp_target,
    active_learner,
    X_full,
    y_full,
    formulas,
    initial_train_idx,
    iterations=20,
    add_per_iter=1,
    color="purple"
):
    """
    Runs an optimization study for a specific number of components (comp_target).
    It resets the learner to the 'initial_train_idx' state (Post-Phase 1)
    before starting, ensuring studies are independent.
    """
    print(f"\n{'='*80}")
    print(f"CASE STUDY: {comp_target}-Component Optimization")
    print(f"Focus: Add points from pool to improve {comp_target}-comp samples specifically.")
    print(f"{'='*80}")

    # 1. Reset Model to End of Phase 1 State
    # (Important so studies don't contaminate each other)
    active_learner.train_indices = initial_train_idx.copy()
    active_learner.X_train = X_full[initial_train_idx].copy()
    active_learner.y_train = y_full[initial_train_idx].copy()
    active_learner._fit_model() # Refit to ensure state is correct

    # 2. Identify Target Validation Set
    comp_counts = np.array([count_components(f) for f in formulas])
    idx_target = np.where(comp_counts == comp_target)[0]

    if len(idx_target) == 0:
        print(f"No {comp_target}-component samples found!")
        return

    print(f"Found {len(idx_target)} {comp_target}-component samples for validation.")

    hist_study = {
        "n_added": [],
        "MAE": [],
        "MAPE": [],
        "NMAE": [],
        "RMSE": [],
        "R2": []
    }

    all_indices = np.arange(len(X_full))

    for i in range(iterations):
        # A. Evaluate on Target Subset ONLY
        y_pred, _ = active_learner.model.predict(X_full[idx_target], return_std=True)
        metrics = ErrorMetrics.compute_all(y_full[idx_target], y_pred)

        hist_study["n_added"].append(i)
        hist_study["MAE"].append(metrics["MAE"])
        hist_study["MAPE"].append(metrics["MAPE"])
        hist_study["NMAE"].append(metrics["NMAE"])
        hist_study["RMSE"].append(metrics["RMSE"])
        hist_study["R2"].append(metrics["R2"])

        print(f"[Iter {i+1}] {comp_target}-Comp MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, NMAE={metrics['NMAE']:.2f}%, R2={metrics['R2']:.4f}")

        # B. Select new points from the REMNANT pool
        remaining_indices = np.setdiff1d(all_indices, active_learner.train_indices)
        X_remaining = X_full[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        sigma_rem, _, _ = active_learner._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_learner.X_train = np.vstack([active_learner.X_train, X_full[select_global_idx]])
        active_learner.y_train = np.concatenate([active_learner.y_train, y_full[select_global_idx]])
        active_learner.train_indices = np.concatenate([active_learner.train_indices, select_global_idx])

        # D. Retrain
        active_learner._fit_model()

    # Final Evaluation
    y_pred_final, y_std_final = active_learner.model.predict(X_full[idx_target], return_std=True)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{comp_target}-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_study["n_added"], hist_study["MAE"], "o-", color=color)
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel(f"MAE on {comp_target}-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot
    ax[1].errorbar(y_full[idx_target], y_pred_final, yerr=y_std_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor=color, markeredgecolor="k")
    ax[1].plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final {comp_target}-Comp Prediction\n(R2={hist_study['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f"Extrapolate{comp_target}comp.png", dpi=300)
    plt.show()

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv')
df4 = pd.read_csv('psp4all.csv')

if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Initial Active Learning (Base Model)
# ------------------------------------------------------------
initial_n = 10
max_samples = 200

np.random.seed(42)
initial_train_indices = np.random.choice(len(X), size=initial_n, replace=False)

print("\n" + "="*80)
print("PHASE 1: Active Learning Base Model")
print("="*80)

active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
model_bm_active, hist_bm_active = active_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    mape_threshold=2.0,
    random_state=42,
)

# Plot Phase 1 Results
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1)", color="#28a99e")

# Store the indices at the end of Phase 1 to use as the starting point for all studies
indices_post_phase1 = active_bm.train_indices.copy()

# ------------------------------------------------------------
# COMPONENT OPTIMIZATION STUDIES (5, 6, 7)
# ------------------------------------------------------------

# Study 1: 5-Component Optimization
run_optimization_study(
    comp_target=5,
    active_learner=active_bm,
    X_full=X,
    y_full=y_bm,
    formulas=formulas,
    initial_train_idx=indices_post_phase1,
    iterations=20,
    add_per_iter=1,
    color="purple"
)

# Study 2: 6-Component Optimization
run_optimization_study(
    comp_target=6,
    active_learner=active_bm,
    X_full=X,
    y_full=y_bm,
    formulas=formulas,
    initial_train_idx=indices_post_phase1,
    iterations=20,
    add_per_iter=1,
    color="darkorange"
)

# Study 3: 7-Component Optimization
run_optimization_study(
    comp_target=7,
    active_learner=active_bm,
    X_full=X,
    y_full=y_bm,
    formulas=formulas,
    initial_train_idx=indices_post_phase1,
    iterations=20,
    add_per_iter=1,
    color="teal"
)

print("\nAll studies completed.")

import re
import numpy as np

def count_components(formula_str):
    """Count unique elements in formula string"""
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula_str))
    components = [elem for elem, count in matches if elem and elem.strip()]
    return len(components)

# Test it:
print(count_components("Al25Nb25Ti25Zr25"))  # → 4
print(count_components("NbMoTiTaVWZr"))      # → 7

def filter_by_components(X, y, formulas, min_comp=1, max_comp=4):
    """Split data by component count"""
    component_counts = np.array([count_components(f) for f in formulas])
    mask = (component_counts >= min_comp) & (component_counts <= max_comp)
    return X[mask], y[mask], formulas[mask], mask, component_counts

# Test it:
#X_train_1_4, y_train_1_4, f_train_1_4, mask, comps = filter_by_components(
#    X, y, formulas, min_comp=1, max_comp=4
#)
#print(f"Training samples (1-4 comp): {len(X_train_1_4)}")

def run_generalization_analysis(X, y, formulas, n_pcs, ErrorMetrics, BayesianExperimentDesign):
    """
    Train on 1-4 components → Predict on 5-7 components
    Returns: R², MAPE, MAE for each component group + plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS: Training on 1-4 Components")
    print("="*80)

    # Split data
    X_train_up4, y_train_up4, formulas_train_up4, mask_train_up4, comp_count_all = \
        filter_by_components(X, y, formulas, min_comp=1, max_comp=4)

    X_test_5, y_test_5, formulas_test_5, _, _ = \
        filter_by_components(X, y, formulas, min_comp=5, max_comp=5)

    X_test_6, y_test_6, formulas_test_6, _, _ = \
        filter_by_components(X, y, formulas, min_comp=6, max_comp=6)

    X_test_7, y_test_7, formulas_test_7, _, _ = \
        filter_by_components(X, y, formulas, min_comp=7, max_comp=7)

    # Report
    print(f"\nTraining: {len(X_train_up4)} samples (1-4 components)")
    print(f"Test 5-comp: {len(X_test_5)}")
    print(f"Test 6-comp: {len(X_test_6)}")
    print(f"Test 7-comp: {len(X_test_7)}")

    if len(X_test_5) + len(X_test_6) + len(X_test_7) == 0:
        print("\n⚠️  No test data with 5-7 components found!")
        return None

    # Train on 1-4
    print(f"\nTraining model on 1-4 component samples...")
    active_gen = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
    model, hist = active_gen.run(
        X_train_up4, y_train_up4, formulas_train_up4,
        initial_indices=None,
        initial_n=10,
        batch_size=1,
        max_samples=200,
        random_state=42
    )

    # Predict on 5-7
    results = {}
    test_configs = [(5, X_test_5, y_test_5), (6, X_test_6, y_test_6), (7, X_test_7, y_test_7)]
    X_test_n = []

    print(f"\nGeneralization Performance:\n")
    for comp_num, X_test, y_test in test_configs:
        if len(X_test) == 0:
            continue

        y_pred, y_std = active_gen.model.predict(X_test, return_std=True)
        metrics = ErrorMetrics.compute_all(y_test, y_pred)

        results[comp_num] = {'y_true': y_test, 'y_pred': y_pred, 'y_std': y_std, 'metrics': metrics}

        print(f"  {comp_num}-component (n={len(X_test)}):")
        print(f"    R²:    {metrics['R2']:.4f}")
        print(f"    MAPE:  {metrics['MAPE']:.2f}%")
        print(f"    NMAE:  {metrics['NMAE']:.2f}%")
        print(f"    MAE:   {metrics['MAE']:.4f}")
        print()
        X_test_n.append(len(X_test))

    # Plot
    n_plots = len(results)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        colors=['purple','darkorange','teal']

        for idx, (comp_num, ax) in enumerate(zip(sorted(results.keys()), axes)):
            r = results[comp_num]
            y_true, y_pred, y_std = r['y_true'], r['y_pred'], r['y_std']

            ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6, capsize=5, c=colors[idx])
            y_lim = [y_true.min(), y_true.max()]
            ax.plot(y_lim, y_lim, 'r--', lw=2, label='Perfect')
            ax.set_xlabel('DFT-computed Bulk Moduli (GPa)', fontweight='bold')
            ax.set_ylabel('ML-predicted Bulk Moduli (GPa)', fontweight='bold')
            ax.set_title(f'{comp_num}-comp\n n={X_test_n[idx]}', fontweight='bold')
            #ax.grid(True, alpha=0.3)
            ax.set_box_aspect(1)
            #ax.legend()

        #plt.tight_layout()
        plt.savefig('generalization_5_7_components.png', dpi=300)
        print(f"✓ Saved: generalization_5_7_components.png")
        plt.show()

    # CSV
    data = [{
        'Components': comp_num,
        'N_Test': len(results[comp_num]['y_true']),
        'R²': f"{results[comp_num]['metrics']['R2']:.4f}",
        'MAPE_%': f"{results[comp_num]['metrics']['MAPE']:.2f}",
        'NMAE_%': f"{results[comp_num]['metrics']['NMAE']:.2f}",
        'MAE': f"{results[comp_num]['metrics']['MAE']:.4f}",
    } for comp_num in sorted(results.keys())]

    df = pd.DataFrame(data)
    df.to_csv('generalization_summary.csv', index=False)
    print(f"✓ Saved: generalization_summary.csv")

    return results

results = run_generalization_analysis(
    X=X,                                    # Your features
    y=y_bm,                                 # Your target (bulk modulus)
    formulas=formulas,                      # Formula strings
    n_pcs=n_pcs,                           # Number of PCs
    ErrorMetrics=ErrorMetrics,
    BayesianExperimentDesign=BayesianExperimentDesign
)



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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true)#np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        # To track indices used in training
        self.train_indices = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, NMAE={metrics['NMAE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAPE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

# ============================================================
# Helper to Identify 5-Component Alloys
# ============================================================
def count_components(formula_str):
    """Counts number of unique elements in a formula string."""
    # Matches any Capital letter followed by optional lowercase letters
    elements = re.findall(r'([A-Z][a-z]*)', str(formula_str))
    return len(set(elements))

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv') # Full 7-comp dataset
df4 = pd.read_csv('psp4all.csv') # 4-comp subset

# Ensure data is loaded
if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Initial Active Learning (Standard)
# ------------------------------------------------------------
initial_n = 10
max_samples = 200

# Fix: Ensure indices are selected from the FULL dataset (X), not df4
np.random.seed(42)
initial_train_indices = np.random.choice(len(X), size=initial_n, replace=False)

print("\n" + "="*80)
print("PHASE 1: Active Learning Base Model")
print("="*80)

# ------------------- ACTIVE SELECTION ------------------------
active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
model_bm_active, hist_bm_active = active_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    mape_threshold=2.0,
    random_state=42,
)

# Plot Phase 1 Results
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1)", color="#28a99e")

# ------------------------------------------------------------
# PHASE 2: Optimization for 6-Component Samples
# ------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 2: Improving 6-Component Predictions")
print("Focus: Add points from pool to improve 5-comp samples specifically.")
print("="*80)

# 1. Identify 6-Component Samples (Validation Set for this phase)
comp_counts = np.array([count_components(f) for f in formulas])
idx_5comp = np.where(comp_counts == 6)[0]

if len(idx_5comp) == 0:
    print("No 6-component samples found! Check formula parsing.")
else:
    print(f"Found {len(idx_5comp)} 6-component samples for validation.")

    # 2. Continue Active Learning Loop
    # We use the existing active_bm object which contains the trained model and X_train

    phase2_iter = 50
    samples_to_add_per_iter = 1

    # History for 6-comp specific metrics
    hist_5comp = {
        "n_added": [],
        "MAE": [],
        "RMSE": [],
        "R2": []
    }

    # All indices
    all_indices = np.arange(len(X))

    for i in range(phase2_iter):
        # A. Evaluate on 5-Component Subset ONLY
        y_pred_5, _ = active_bm.model.predict(X[idx_5comp], return_std=True)
        metrics_5 = ErrorMetrics.compute_all(y_bm[idx_5comp], y_pred_5)

        hist_5comp["n_added"].append(i)
        hist_5comp["MAE"].append(metrics_5["MAE"])
        hist_5comp["RMSE"].append(metrics_5["RMSE"])
        hist_5comp["R2"].append(metrics_5["R2"])

        print(f"[PHASE 2] Iter {i+1}: 6-Comp MAE={metrics_5['MAE']:.4f}, R2={metrics_5['R2']:.4f}")

        # B. Select new points from the REMNANT pool (Full pool - Trained)
        remaining_indices = np.setdiff1d(all_indices, active_bm.train_indices)
        X_remaining = X[remaining_indices]
        y_remaining = y_bm[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        # This helps the GP learn the global landscape better, indirectly helping 5-comp
        sigma_rem, _, _ = active_bm._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-samples_to_add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_bm.X_train = np.vstack([active_bm.X_train, X[select_global_idx]])
        active_bm.y_train = np.concatenate([active_bm.y_train, y_bm[select_global_idx]])
        active_bm.train_indices = np.concatenate([active_bm.train_indices, select_global_idx])

        # D. Retrain
        active_bm._fit_model()

    # Final Evaluation on 5-Component
    y_pred_5_final, y_std_5_final = active_bm.model.predict(X[idx_5comp], return_std=True)

    # Plot Phase 2 Improvement
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Phase 2: 6-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_5comp["n_added"], hist_5comp["MAE"], "o-", color="purple")
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel("MAE on 6-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot for 5-Comp
    ax[1].errorbar(y_bm[idx_5comp], y_pred_5_final, yerr=y_std_5_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor="purple", markeredgecolor="k")
    ax[1].plot([y_bm.min(), y_bm.max()], [y_bm.min(), y_bm.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final 6-Comp Prediction\n(R2={hist_5comp['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.show()

print("\nSaved metrics_comparison_bulk_only.csv")
#active_bm = active_bm_copy

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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.mean(y_true)#np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        # To track indices used in training
        self.train_indices = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["NMAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["NMAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("NMAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

# ============================================================
# Helper to Identify 5-Component Alloys
# ============================================================
def count_components(formula_str):
    """Counts number of unique elements in a formula string."""
    # Matches any Capital letter followed by optional lowercase letters
    elements = re.findall(r'([A-Z][a-z]*)', str(formula_str))
    return len(set(elements))

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv') # Full 7-comp dataset
df4 = pd.read_csv('psp4all.csv') # 4-comp subset

# Ensure data is loaded
if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Initial Active Learning (Standard)
# ------------------------------------------------------------
initial_n = 10
max_samples = 200

# Fix: Ensure indices are selected from the FULL dataset (X), not df4
np.random.seed(42)
initial_train_indices = np.random.choice(len(X), size=initial_n, replace=False)

print("\n" + "="*80)
print("PHASE 1: Active Learning Base Model")
print("="*80)

active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
model_bm_active, hist_bm_active = active_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    mape_threshold=2.0,
    random_state=42,
)

# Plot Phase 1 Results
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1)", color="#28a99e")

# ------------------------------------------------------------
# PHASE 2: Optimization for 5-Component Samples
# ------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 2: \n Improving 5-Component Predictions")
print("Focus: Add points from pool to improve 5-comp samples specifically.")
print("="*80)

# 1. Identify 5-Component Samples (Validation Set for this phase)
comp_counts = np.array([count_components(f) for f in formulas])
idx_5comp = np.where(comp_counts == 5)[0]

if len(idx_5comp) == 0:
    print("No 5-component samples found! Check formula parsing.")
else:
    print(f"Found {len(idx_5comp)} 5-component samples for validation.")

    # 2. Continue Active Learning Loop
    # We use the existing active_bm object which contains the trained model and X_train

    phase2_iter = 20
    samples_to_add_per_iter = 1

    # History for 5-comp specific metrics
    hist_5comp = {
        "n_added": [],
        "MAE": [],
        "RMSE": [],
        "R2": []
    }

    # All indices
    all_indices = np.arange(len(X))

    for i in range(phase2_iter):
        # A. Evaluate on 5-Component Subset ONLY
        y_pred_5, _ = active_bm.model.predict(X[idx_5comp], return_std=True)
        metrics_5 = ErrorMetrics.compute_all(y_bm[idx_5comp], y_pred_5)

        hist_5comp["n_added"].append(i)
        hist_5comp["MAE"].append(metrics_5["MAE"])
        hist_5comp["RMSE"].append(metrics_5["RMSE"])
        hist_5comp["R2"].append(metrics_5["R2"])

        print(f"[PHASE 2] Iter {i+1}: 5-Comp MAE={metrics_5['MAE']:.4f}, R2={metrics_5['R2']:.4f}")

        # B. Select new points from the REMNANT pool (Full pool - Trained)
        remaining_indices = np.setdiff1d(all_indices, active_bm.train_indices)
        X_remaining = X[remaining_indices]
        y_remaining = y_bm[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        # This helps the GP learn the global landscape better, indirectly helping 5-comp
        sigma_rem, _, _ = active_bm._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-samples_to_add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_bm.X_train = np.vstack([active_bm.X_train, X[select_global_idx]])
        active_bm.y_train = np.concatenate([active_bm.y_train, y_bm[select_global_idx]])
        active_bm.train_indices = np.concatenate([active_bm.train_indices, select_global_idx])

        # D. Retrain
        active_bm._fit_model()

    # Final Evaluation on 5-Component
    y_pred_5_final, y_std_5_final = active_bm.model.predict(X[idx_5comp], return_std=True)

    # Plot Phase 2 Improvement
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Phase 2: 5-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_5comp["n_added"], hist_5comp["MAE"], "o-", color="purple")
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel("MAE on 5-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot for 5-Comp
    ax[1].errorbar(y_bm[idx_5comp], y_pred_5_final, yerr=y_std_5_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor="purple", markeredgecolor="k")
    ax[1].plot([y_bm.min(), y_bm.max()], [y_bm.min(), y_bm.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final 5-Comp Prediction\n(R2={hist_5comp['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.show()

print("\nSaved metrics_comparison_bulk_only.csv")

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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "NMAE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        # To track indices used in training
        self.train_indices = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["NMAE"].append(metrics["NMAE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["NMAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("NMAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

# ============================================================
# Helper to Identify 5-Component Alloys
# ============================================================
def count_components(formula_str):
    """Counts number of unique elements in a formula string."""
    # Matches any Capital letter followed by optional lowercase letters
    elements = re.findall(r'([A-Z][a-z]*)', str(formula_str))
    return len(set(elements))

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv') # Full 7-comp dataset
df4 = pd.read_csv('psp4all.csv') # 4-comp subset

# Ensure data is loaded
if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Initial Active Learning (Standard)
# ------------------------------------------------------------
initial_n = 10
max_samples = 200

# Fix: Ensure indices are selected from the FULL dataset (X), not df4
np.random.seed(42)
initial_train_indices = np.random.choice(len(X), size=initial_n, replace=False)

print("\n" + "="*80)
print("PHASE 1: Active Learning Base Model")
print("="*80)


# Plot Phase 1 Results
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1)", color="#28a99e")

# ------------------------------------------------------------
# PHASE 2: Optimization for 5-Component Samples
# ------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 2: Improving 5-Component Predictions")
print("Focus: Add points from pool to improve 5-comp samples specifically.")
print("="*80)

# 1. Identify 5-Component Samples (Validation Set for this phase)
comp_counts = np.array([count_components(f) for f in formulas])
idx_5comp = np.where(comp_counts == 5)[0]

if len(idx_5comp) == 0:
    print("No 5-component samples found! Check formula parsing.")
else:
    print(f"Found {len(idx_5comp)} 5-component samples for validation.")

    # 2. Continue Active Learning Loop
    # We use the existing active_bm object which contains the trained model and X_train

    phase2_iter = 50
    samples_to_add_per_iter = 1

    # History for 5-comp specific metrics
    hist_5comp = {
        "n_added": [],
        "MAE": [],
        "MAPE": [],
        "RMSE": [],
        "R2": []
    }

    # All indices
    all_indices = np.arange(len(X))

    for i in range(phase2_iter):
        # A. Evaluate on 5-Component Subset ONLY
        y_pred_5, _ = active_bm.model.predict(X[idx_5comp], return_std=True)
        metrics_5 = ErrorMetrics.compute_all(y_bm[idx_5comp], y_pred_5)

        hist_5comp["n_added"].append(i)
        hist_5comp["MAE"].append(metrics_5["MAE"])
        hist_5comp["MAPE"].append(metrics_5["MAPE"])
        hist_5comp["RMSE"].append(metrics_5["RMSE"])
        hist_5comp["R2"].append(metrics_5["R2"])

        print(f"[PHASE 2] Iter {i+1}: 5-Comp MAE={metrics_5['MAE']:.4f}, MAPE={metrics_5['MAPE']:.2f}%, R2={metrics_5['R2']:.4f}")

        # B. Select new points from the REMNANT pool (Full pool - Trained)
        remaining_indices = np.setdiff1d(all_indices, active_bm.train_indices)
        X_remaining = X[remaining_indices]
        y_remaining = y_bm[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        # This helps the GP learn the global landscape better, indirectly helping 5-comp
        sigma_rem, _, _ = active_bm._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-samples_to_add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_bm.X_train = np.vstack([active_bm.X_train, X[select_global_idx]])
        active_bm.y_train = np.concatenate([active_bm.y_train, y_bm[select_global_idx]])
        active_bm.train_indices = np.concatenate([active_bm.train_indices, select_global_idx])

        # D. Retrain
        active_bm._fit_model()

    # Final Evaluation on 5-Component
    y_pred_5_final, y_std_5_final = active_bm.model.predict(X[idx_5comp], return_std=True)

    # Plot Phase 2 Improvement
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Phase 2: 5-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_5comp["n_added"], hist_5comp["MAE"], "o-", color="purple")
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel("MAE on 5-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot for 5-Comp
    ax[1].errorbar(y_bm[idx_5comp], y_pred_5_final, yerr=y_std_5_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor="purple", markeredgecolor="k")
    ax[1].plot([y_bm.min(), y_bm.max()], [y_bm.min(), y_bm.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final 5-Comp Prediction\n(R2={hist_5comp['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.show()

print("\nSaved metrics_comparison_bulk_only.csv")

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
# Error metrics (sklearn + custom NMAE)
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE calculation with epsilon for near-zero denominators."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        epsilon = 1e-6 * np.max(np.abs(y_true))
        mape_val = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        return float(np.mean(mape_val) * 100.0)

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.max(y_true) - np.min(y_true)
        if rng == 0:
            return 0.0
        return float(np.abs(100.0 * mean_absolute_error(y_true, y_pred) / rng))

    @staticmethod
    def r_squared(y_true, y_pred):
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def median_ae(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_all(y_true, y_pred):
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
            "Median_AE": ErrorMetrics.median_ae(y_true, y_pred),
        }

# ============================================================
# GPyTorch Gaussian Process with ARDSE kernel
# ============================================================

def create_ard_rbf_model(train_x, train_y, likelihood, n_pcs):
    class ARDRBFModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ARDRBFModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = LCMKernel(base_kernels=[RBFKernel(ard_num_dims=n_pcs)], num_tasks=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    return ARDRBFModel(train_x, train_y, likelihood)

class GPyTorchGPR_ARDSE:
    def __init__(self, n_pcs, alpha=1e-6, device="cpu", ardse=None):
        self.n_pcs = n_pcs
        self.alpha = alpha
        self.device = DEVICE
        self.model = None
        self.likelihood = None
        self._fitted = False
        self.ardse = ardse
        self.scaler_y = StandardScaler()

    def fit(self, X, y, n_epochs=100, lr=0.1, verbose=False):
        Xs = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        ys = self.scaler_y.fit_transform(y).ravel()

        X_train = torch.from_numpy(Xs).float().to(self.device)
        y_train = torch.from_numpy(ys).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.model = create_ard_rbf_model(X_train, y_train, self.likelihood, self.n_pcs)

        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        Xs = np.asarray(X)
        X_test = torch.from_numpy(Xs).float().to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_test))
            y_mean_s = preds.mean.cpu().numpy()
            y_std_s = preds.stddev.cpu().numpy()

        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]

        if return_std:
            return y_mean, y_std
        return y_mean

# ============================================================
# Active learning: Bayesian experiment design
# ============================================================

class BayesianExperimentDesign:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.model = None
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        # To track indices used in training
        self.train_indices = None

    def _fit_model(self):
        self.model = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
        self.model.fit(self.X_train, self.y_train, n_epochs=200, lr=0.1, verbose=False)

    def _information_gain(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, mape_threshold=2.0, random_state=42):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        # --- INITIALIZATION ---
        if initial_indices is not None:
            train_indices = np.array(initial_indices)
        else:
            elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
            elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas])

            if np.sum(elemental_mask) >= 4:
                train_indices = np.where(elemental_mask)[0][:4]
            else:
                print("Warning: Could not find 4 elemental compositions. Using random initialization.")
                np.random.seed(random_state)
                train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)

        self.X_train = X_all[train_indices].copy()
        self.y_train = y_all[train_indices].copy()
        self.train_indices = train_indices

        print(f"Initial training set: {len(self.X_train)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(self.X_train) < len(X_all):
            self._fit_model()

            remaining_indices = np.setdiff1d(all_indices, self.train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = self.model.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(self.X_train) >= max_samples:
                break

            ig_remaining, _, _ = self._information_gain(X_remaining)
            select_local_idx = np.argsort(ig_remaining)[-batch_size:]
            select_global_idx = remaining_indices[select_local_idx]

            self.X_train = np.vstack([self.X_train, X_all[select_global_idx]])
            self.y_train = np.concatenate([self.y_train, y_all[select_global_idx]])
            self.train_indices = np.concatenate([self.train_indices, select_global_idx])

            iteration += 1

        self._fit_model()
        return self.model, self.history

# ============================================================
# Random sampling baseline
# ============================================================

class RandomSamplingBaseline:
    def __init__(self, n_pcs, kernel="ardse"):
        self.n_pcs = n_pcs
        self.kernel = kernel
        self.history = {
            "n_samples": [],
            "MAE": [],
            "MAPE": [],
            "RMSE": [],
            "R2": [],
            "MeanSigma": [],
        }
        self.model = None

    def run(self, X_all, y_all, formulas, initial_indices=None, initial_n=4, batch_size=1,
            max_samples=35, random_state=123):

        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)
        all_indices = np.arange(len(X_all))

        if initial_indices is not None:
            train_indices = np.array(initial_indices)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()
        else:
            np.random.seed(random_state)
            train_indices = np.random.choice(len(X_all), size=min(initial_n, len(X_all)), replace=False)
            X_train = X_all[train_indices].copy()
            y_train = y_all[train_indices].copy()

        iteration = 0
        while len(X_train) <= max_samples and len(X_train) < len(X_all):
            gpr = GPyTorchGPR_ARDSE(n_pcs=self.n_pcs, ardse=self.kernel)
            gpr.fit(X_train, y_train, n_epochs=200, lr=0.1, verbose=False)

            remaining_indices = np.setdiff1d(all_indices, train_indices)
            X_remaining = X_all[remaining_indices]
            y_remaining = y_all[remaining_indices]

            y_pred_remaining, y_std_remaining = gpr.predict(X_remaining, return_std=True)
            metrics = ErrorMetrics.compute_all(y_remaining, y_pred_remaining)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_remaining)))

            print(f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                  f"n_test={len(remaining_indices):4d}, MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}")

            if len(remaining_indices) == 0 or len(X_train) >= max_samples:
                break

            k = min(batch_size, len(remaining_indices))
            np.random.seed(random_state + iteration)
            select_local_idx = np.random.choice(np.arange(len(remaining_indices)), size=k, replace=False)
            select_global_idx = remaining_indices[select_local_idx]

            X_train = np.vstack([X_train, X_all[select_global_idx]])
            y_train = np.concatenate([y_train, y_all[select_global_idx]])
            train_indices = np.concatenate([train_indices, select_global_idx])

            iteration += 1

        self.model = gpr
        return self.model, self.history

# ============================================================
# Plotting Helpers
# ============================================================
def plot_figure3_pca_variance(df, pc_cols, fname="Figure_3_PCA_Variance.png"):
    pcs = df[pc_cols].values
    var = np.var(pcs, axis=0, ddof=1)
    var_ratio = var / np.sum(var)
    n_show = min(50, len(pc_cols))
    vals = var_ratio[:n_show] * 100.0
    cum = np.cumsum(vals)

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
    plt.show()

def plot_figure5_pc_space(df, pc_cols, n, target_col="bulk_modulus", fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 1)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def pcquad(df, pc_cols, n, target_col="bulk_modulus", fname="pcquad.png"):
    if len(pc_cols) < 3: return
    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(2, 2, 4)
    sc1 = ax1.scatter(pc2, pc3, c=target, cmap="viridis", s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[1], fontweight="bold")
    ax1.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax2 = fig.add_subplot(2, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target, cmap="rainbow", s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Bulk Modulus (GPa)', fontsize=9)

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(df['PC1'], df['PC2'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    layers = [
        (f'Al',  df['Al_pct'] > n, 'red',    'o'),
        (f'Ti',  df['Ti_pct'] > n, 'blue',   'o'),
        (f'Zr',  df['Zr_pct'] > n, 'green',  'o'),
        (f'Nb',  df['Nb_pct'] > n, 'orange', 'o')
    ]

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            ax0.scatter(subset['PC1'], subset['PC2'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    ax0.set_xlabel(pc_cols[0], fontweight="bold")
    ax0.set_ylabel(pc_cols[1], fontweight="bold")

    axx = fig.add_subplot(2, 2, 3)
    axx.scatter(df['PC2'], df['PC3'], c='lightgray', label='All Data', s=100, alpha=0.5, zorder=1)

    for label, mask, color, marker in layers:
        subset = df[mask]
        if not subset.empty:
            axx.scatter(subset['PC2'], subset['PC3'], c=color, label=label, marker=marker,
                        s=100, alpha=0.8, edgecolors='k', linewidth=0.5, zorder=2)

    axx.set_xlabel(pc_cols[1], fontweight="bold")
    axx.set_ylabel(pc_cols[2], fontweight="bold")

    ax0.set_box_aspect(1)
    axx.set_box_aspect(1)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.show()

def plot_convergence(active_hist, random_hist, property_name, fname="Figure_6_Convergence.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # NMAE
    ax = axes[0, 0]
    ax.axhline(y=2, color='grey', linestyle='--', linewidth=2)
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAPE (%)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R2
    ax = axes[0, 1]
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("R²", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1, 0]
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("MAE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-", label="Random", color="maroon", linewidth=2)
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-", label="Active", color="limegreen", linewidth=2)
    ax.set_title("RMSE", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, y_std, property_name, fname="Figure_7_Predictions.png", color=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    ax = axes[0]
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 'k--', linewidth=2)

    c = color if color else "#2E86AB"
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray", capsize=3, alpha=0.7,
                markersize=5, markeredgecolor="k", markerfacecolor=c)
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_box_aspect(1)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k", facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    band = 2.0 * np.mean(y_std)
    ax.axhspan(-band, band, color="gray", alpha=0.2, label="±2·mean(σ)")
    ax.set_xlabel("Prediction", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

def plot_uncertainty_hist(y_true, y_pred, y_std, property_name, fname="Figure_7b_Uncertainty.png", color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    c = color if color else "#2E86AB"
    ax.hist(y_std, bins=30, color=c, alpha=0.8, edgecolor="k")
    ax.axvline(np.mean(y_std), color="r", linestyle="--", label=f"mean σ = {np.mean(y_std):.3f}")
    ax.set_xlabel("Predicted σ", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(f"Uncertainty Distribution ({property_name})", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

def process_alloy_data(df, output_csv_path=None):
    def parse_formula(formula):
        matches = re.findall(r'([A-Z][a-z]*)(\d+)', str(formula))
        return {elem: int(count) for elem, count in matches}

    parsed_counts = df['formula'].apply(parse_formula)
    df_counts = pd.DataFrame(parsed_counts.tolist())
    df_counts = df_counts.fillna(0).astype(int)

    target_elements = ['Al', 'Nb', 'Ti', 'Zr']
    for elem in target_elements:
        if elem not in df_counts.columns:
            df_counts[elem] = 0

    df_counts = df_counts[target_elements]
    df_final = pd.concat([df, df_counts], axis=1)

    for elem in target_elements:
        df_final[f'{elem}_pct'] = (df_final[elem] / 128) * 100

    if output_csv_path:
        df_final.to_csv(output_csv_path, index=False)
    return df_final

# ============================================================
# Helper to Identify 5-Component Alloys
# ============================================================
def count_components(formula_str):
    """Counts number of unique elements in a formula string."""
    # Matches any Capital letter followed by optional lowercase letters
    elements = re.findall(r'([A-Z][a-z]*)', str(formula_str))
    return len(set(elements))

# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv') # Full 7-comp dataset
df4 = pd.read_csv('psp4all.csv') # 4-comp subset

# Ensure data is loaded
if len(df) < 1000:
    print("Looking for full dataset...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)
df4 = df4.reset_index(drop=True)

# Features and Targets
pc_cols = [c for c in df.columns if c.startswith("PC")]
n_pcs = 3
X_raw = df[pc_cols[:n_pcs]].values
y_bm  = df["bulk_modulus"].values

# GLOBAL Scaling
scaler_X_global = StandardScaler()
X = scaler_X_global.fit_transform(X_raw)

# Get formulas for the FULL dataset
formulas = df['formula'].values

print(f"✓ Loaded dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")

# Process elemental data for plotting
df_final = process_alloy_data(df)
n = 75
if len(df_final) >= 100:
    plot_figure5_pc_space(df_final, pc_cols, n=98, target_col="bulk_modulus")
    pcquad(df_final, pc_cols, n=n, target_col="bulk_modulus")

# ------------------------------------------------------------
# PHASE 1: Initial Active Learning (Standard)
# ------------------------------------------------------------
initial_n = 10
max_samples = 200

# Fix: Ensure indices are selected from the FULL dataset (X), not df4
np.random.seed(42)
initial_train_indices = np.random.choice(len(X), size=initial_n, replace=False)

print("\n" + "="*80)
print("PHASE 1: Active Learning Base Model")
print("="*80)

# ------------------- ACTIVE SELECTION ------------------------
active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
model_bm_active, hist_bm_active = active_bm.run(
    X, y_bm, formulas,
    initial_indices=initial_train_indices,
    initial_n=initial_n,
    batch_size=1,
    max_samples=max_samples,
    mape_threshold=2.0,
    random_state=42,
)

# Plot Phase 1 Results
y_bm_pred, y_bm_std = active_bm.model.predict(X, return_std=True)
plot_predictions(y_bm, y_bm_pred, y_bm_std, "Bulk Modulus (Phase 1)", color="#28a99e")

# ------------------------------------------------------------
# PHASE 2: Optimization for 6-Component Samples
# ------------------------------------------------------------
print("\n" + "="*80)
print("PHASE 2: Improving 7-Component Predictions")
print("Focus: Add points from pool to improve 7-comp samples specifically.")
print("="*80)

# 1. Identify 6-Component Samples (Validation Set for this phase)
comp_counts = np.array([count_components(f) for f in formulas])
idx_5comp = np.where(comp_counts == 7)[0]

if len(idx_5comp) == 0:
    print("No 7-component samples found! Check formula parsing.")
else:
    print(f"Found {len(idx_5comp)} 7-component samples for validation.")

    # 2. Continue Active Learning Loop
    # We use the existing active_bm object which contains the trained model and X_train

    phase2_iter = 50
    samples_to_add_per_iter = 1

    # History for 7-comp specific metrics
    hist_5comp = {
        "n_added": [],
        "MAE": [],
        "MAPE": [],
        "RMSE": [],
        "R2": []
    }

    # All indices
    all_indices = np.arange(len(X))

    for i in range(phase2_iter):
        # A. Evaluate on 5-Component Subset ONLY
        y_pred_5, _ = active_bm.model.predict(X[idx_5comp], return_std=True)
        metrics_5 = ErrorMetrics.compute_all(y_bm[idx_5comp], y_pred_5)

        hist_5comp["n_added"].append(i)
        hist_5comp["MAE"].append(metrics_5["MAE"])
        hist_5comp["MAPE"].append(metrics_5["MAPE"])
        hist_5comp["RMSE"].append(metrics_5["RMSE"])
        hist_5comp["R2"].append(metrics_5["R2"])

        print(f"[PHASE 2] Iter {i+1}: 7-Comp MAE={metrics_5['MAE']:.4f}, R2={metrics_5['R2']:.4f}")

        # B. Select new points from the REMNANT pool (Full pool - Trained)
        remaining_indices = np.setdiff1d(all_indices, active_bm.train_indices)
        X_remaining = X[remaining_indices]
        y_remaining = y_bm[remaining_indices]

        if len(remaining_indices) == 0:
            print("Pool exhausted.")
            break

        # Use Max Sigma (Uncertainty Sampling) on the WHOLE pool
        # This helps the GP learn the global landscape better, indirectly helping 5-comp
        sigma_rem, _, _ = active_bm._information_gain(X_remaining)

        # Select highest uncertainty
        select_local_idx = np.argsort(sigma_rem)[-samples_to_add_per_iter:]
        select_global_idx = remaining_indices[select_local_idx]

        # C. Update Model Data
        active_bm.X_train = np.vstack([active_bm.X_train, X[select_global_idx]])
        active_bm.y_train = np.concatenate([active_bm.y_train, y_bm[select_global_idx]])
        active_bm.train_indices = np.concatenate([active_bm.train_indices, select_global_idx])

        # D. Retrain
        active_bm._fit_model()

    # Final Evaluation on 5-Component
    y_pred_5_final, y_std_5_final = active_bm.model.predict(X[idx_5comp], return_std=True)

    # Plot Phase 2 Improvement
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Phase 2: 7-Component Optimization", fontweight="bold")

    # Subplot 1: MAE reduction
    ax[0].plot(hist_5comp["n_added"], hist_5comp["MAE"], "o-", color="purple")
    ax[0].set_xlabel("Additional Samples Added")
    ax[0].set_ylabel("MAE on 7-Comp Subset")
    ax[0].set_title("Error Reduction")
    ax[0].grid(True, alpha=0.3)

    # Subplot 2: Final Parity Plot for 5-Comp
    ax[1].errorbar(y_bm[idx_5comp], y_pred_5_final, yerr=y_std_5_final, fmt="o",
                   ecolor="gray", alpha=0.6, markerfacecolor="purple", markeredgecolor="k")
    ax[1].plot([y_bm.min(), y_bm.max()], [y_bm.min(), y_bm.max()], 'k--')
    ax[1].set_xlabel("Ground Truth")
    ax[1].set_ylabel("Prediction")
    ax[1].set_title(f"Final 7-Comp Prediction\n(R2={hist_5comp['R2'][-1]:.3f})")
    ax[1].set_box_aspect(1)

    plt.tight_layout()
    plt.show()

print("\nSaved metrics_comparison_bulk_only.csv")

import re
import numpy as np


def run_generalization_analysis2(X, y, formulas, n_pcs, ErrorMetrics, BayesianExperimentDesign):
    """
    Train on 1-4 components → Predict on 5-7 components
    Returns: R², MAPE, MAE for each component group + plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS: Training on 1-4 Components")
    print("="*80)

    # Split data
    X_train_up4, y_train_up4, formulas_train_up4, mask_train_up4, comp_count_all = \
        filter_by_components(X, y, formulas, min_comp=1, max_comp=5)

    #X_test_5, y_test_5, formulas_test_5, _, _ = \
        #filter_by_components(X, y, formulas, min_comp=6, max_comp=5)

    X_test_6, y_test_6, formulas_test_6, _, _ = \
        filter_by_components(X, y, formulas, min_comp=6, max_comp=6)

    X_test_7, y_test_7, formulas_test_7, _, _ = \
        filter_by_components(X, y, formulas, min_comp=7, max_comp=7)

    # Report
    print(f"\nTraining: {len(X_train_up4)} samples (1-5 components)")
    #print(f"Test 5-comp: {len(X_test_5)}")
    print(f"Test 6-comp: {len(X_test_6)}")
    print(f"Test 7-comp: {len(X_test_7)}")

    if len(X_test_6) + len(X_test_7) == 0:
        print("\n⚠️  No test data with 5-7 components found!")
        return None

    # Train on 1-4
    print(f"\nTraining model on 1-4 component samples...")
    active_gen = BayesianExperimentDesign(n_pcs=n_pcs, kernel="ardse")
    model, hist = active_gen.run(
        X_train_up4, y_train_up4, formulas_train_up4,
        initial_indices=None,
        initial_n=10,
        batch_size=1,
        max_samples=300,
        random_state=42
    )

    # Predict on 5-7
    results = {}
    test_configs = [(6, X_test_6, y_test_6), (7, X_test_7, y_test_7)]

    print(f"\nGeneralization Performance:\n")
    for comp_num, X_test, y_test in test_configs:
        if len(X_test) == 0:
            continue

        y_pred, y_std = active_gen.model.predict(X_test, return_std=True)
        metrics = ErrorMetrics.compute_all(y_test, y_pred)

        results[comp_num] = {'y_true': y_test, 'y_pred': y_pred, 'y_std': y_std, 'metrics': metrics}

        print(f"  {comp_num}-component (n={len(X_test)}):")
        print(f"    R²:    {metrics['R2']:.4f}")
        print(f"    MAPE:  {metrics['MAPE']:.2f}%")
        print(f"    MAE:   {metrics['MAE']:.4f}")
        print()

    # Plot
    n_plots = len(results)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for idx, (comp_num, ax) in enumerate(zip(sorted(results.keys()), axes)):
            r = results[comp_num]
            y_true, y_pred, y_std = r['y_true'], r['y_pred'], r['y_std']

            ax.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.6, capsize=5)
            y_lim = [y_true.min(), y_true.max()]
            ax.plot(y_lim, y_lim, 'r--', lw=2, label='Perfect')
            ax.set_xlabel('True (GPa)', fontweight='bold')
            ax.set_ylabel('Predicted (GPa)', fontweight='bold')
            ax.set_title(f'{comp_num}-comp\nR²={r["metrics"]["R2"]:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig('generalization_6_7_components.png', dpi=300)
        print(f"✓ Saved: generalization_6_7_components.png")
        plt.show()

    # CSV
    data = [{
        'Components': comp_num,
        'N_Test': len(results[comp_num]['y_true']),
        'R²': f"{results[comp_num]['metrics']['R2']:.4f}",
        'MAPE_%': f"{results[comp_num]['metrics']['MAPE']:.2f}",
        'MAE': f"{results[comp_num]['metrics']['MAE']:.4f}",
    } for comp_num in sorted(results.keys())]

    df = pd.DataFrame(data)
    df.to_csv('generalization67_summary.csv', index=False)
    print(f"✓ Saved: generalization67_summary.csv")

    return results

results = run_generalization_analysis2(
    X=X,                                    # Your features
    y=y_bm,                                 # Your target (bulk modulus)
    formulas=formulas,                      # Formula strings
    n_pcs=n_pcs,                           # Number of PCs
    ErrorMetrics=ErrorMetrics,
    BayesianExperimentDesign=BayesianExperimentDesign
)



df = pd.read_csv('pca_pspall7.csv') #pca_pspall7 #pca_psp7

# If this is a small stub file, try to find the full one
if len(df) < 1000:
    print("Looking for full dataset with 4495 samples...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        print(f"Found full dataset: {csv_path} ({os.path.getsize(csv_path)} bytes)")
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)

import numpy as np
import pandas as pd
import torch
import gpytorch
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

# ==========================================
# 1. DATA PREPARATION & ALIGNMENT
# ==========================================

def get_component_count(formula):
    elements = re.findall(r'[A-Z][a-z]*', str(formula))
    return len(set(elements))

def load_data(psp4all_path, pca_pspall7_path):
    """
    Loads data and aligns psp4all formulas with pca_pspall7 features.
    """
    # Load raw csvs
    df_4comp_list = pd.read_csv(psp4all_path) # Contains list of 4-comp formulas
    df_master = pd.read_csv(pca_pspall7_path) # Contains Features (PCs) for everything

    # Filter valid
    df_master = df_master.dropna(subset=['bulk_modulus', 'formula'])

    # Extract Feature Matrix and Target
    feature_cols = [f'PC{i}' for i in range(1, 51)]

    scaler_x = StandardScaler()
    X_raw = scaler_x.fit_transform(df_master[feature_cols].values)

    scaler_y = StandardScaler()
    y_raw = scaler_y.fit_transform(df_master['bulk_modulus'].values.reshape(-1, 1)).flatten()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X_raw, dtype=torch.float32).to(device)
    y = torch.tensor(y_raw, dtype=torch.float32).to(device)

    # --- INDICES SETUP ---
    # 1. Identify indices for the "4-Component Dataset" (psp4all)
    # We map formulas from psp4all to the master dataframe
    formulas_4comp = set(df_4comp_list['formula'].unique())
    is_4comp = df_master['formula'].isin(formulas_4comp)

    indices_4comp_total = np.where(is_4comp)[0].tolist()

    # 2. Identify indices for Validation Sets
    # Note: validation sets overlap with the pool, that's fine for tracking metrics.
    df_master['n_comp'] = df_master['formula'].apply(get_component_count)

    indices = {
        '4comp_total': indices_4comp_total,
        'global_all': np.arange(len(df_master)).tolist(),
        'val_all': np.arange(len(df_master)).tolist(),
        'val_5': np.where(df_master['n_comp'] == 5)[0].tolist(),
        'val_6': np.where(df_master['n_comp'] == 6)[0].tolist(),
        'val_7': np.where(df_master['n_comp'] == 7)[0].tolist()
    }

    print(f"Total Data: {len(df_master)}")
    print(f"Total 4-Comp Samples (psp4all): {len(indices_4comp_total)}")

    return X, y, indices, scaler_y

# ==========================================
# 2. MODEL DEFINITION
# ==========================================

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def train_gp(train_x, train_y, training_iter=50):
    device = train_x.device
    likelihood = GaussianLikelihood().to(device)
    model = GPModel(train_x, train_y, likelihood).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood

def predict(model, likelihood, test_x):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if test_x.numel() == 0: return torch.tensor([]), torch.tensor([])
        preds = likelihood(model(test_x))
    return preds.mean, preds.variance

# ==========================================
# 3. STAGE 1: ACTIVE LEARNING ON 4-COMP
# ==========================================

def run_stage_1_initialization(X, y, indices_4comp, start_size=10, steps=20):
    """
    Actively learns a model purely within the 4-component space
    to create the 'Initial Model'.
    """
    print(f"\n[Stage 1] Initializing Model on 4-Component Data...")

    # 1. Start with random subset of 4-comp
    np.random.seed(42)
    start_indices = np.random.choice(indices_4comp, size=start_size, replace=False).tolist()

    train_idx = list(start_indices)
    # Pool is the REST of the 4-comp data
    pool_idx = list(set(indices_4comp) - set(train_idx))

    # AL Loop (Simplified for Stage 1: Just get the best samples)
    for i in range(steps):
        # Train
        X_train = X[train_idx]
        y_train = y[train_idx]
        model, likelihood = train_gp(X_train, y_train, training_iter=30)

        # Select from 4-comp Pool
        X_pool = X[pool_idx]
        _, pool_vars = predict(model, likelihood, X_pool)

        best_local = torch.argmax(pool_vars).item()
        best_global = pool_idx[best_local]

        train_idx.append(best_global)
        pool_idx.pop(best_local)

        if i % 5 == 0:
            print(f"  Stage 1 Iter {i}/{steps}: Train Size {len(train_idx)}")

    print(f"[Stage 1] Complete. Final Training Set Size: {len(train_idx)}")
    return train_idx

# ==========================================
# 4. STAGE 2: ACTIVE LEARNING ON GLOBAL POOL
# ==========================================

def run_stage_2_extrapolation(X, y, initial_train_idx, indices, scaler_y, steps=50):
    """
    Starts with the Stage 1 model.
    Pool = EVERYTHING not in initial_train_idx.
    Tracks performance on 4 Case Studies.
    """
    print(f"\n[Stage 2] Extrapolating to Global Space...")

    train_idx = list(initial_train_idx)
    # Global Pool: All data indices minus current training
    # This includes remaining 4-comp + 1,2,3,5,6,7 comp
    all_indices = set(indices['global_all'])
    pool_idx = list(all_indices - set(train_idx))

    # Metrics Storage
    history = {
        'All (1-7)': {'RMSE': [], 'R2': []},
        '5-Comp':    {'RMSE': [], 'R2': []},
        '6-Comp':    {'RMSE': [], 'R2': []},
        '7-Comp':    {'RMSE': [], 'R2': []}
    }

    for i in range(steps + 1):
        # 1. Train
        X_train = X[train_idx]
        y_train = y[train_idx]
        model, likelihood = train_gp(X_train, y_train, training_iter=30)

        # 2. Evaluate Case Studies
        case_map = {
            'All (1-7)': indices['val_all'],
            '5-Comp':    indices['val_5'],
            '6-Comp':    indices['val_6'],
            '7-Comp':    indices['val_7']
        }

        for name, val_idx in case_map.items():
            if len(val_idx) == 0: continue

            # Predict
            mu, _ = predict(model, likelihood, X[val_idx])

            # Inverse Transform
            true_real = scaler_y.inverse_transform(y[val_idx].cpu().reshape(-1,1)).flatten()
            pred_real = scaler_y.inverse_transform(mu.cpu().reshape(-1,1)).flatten()

            # Store
            rmse = np.sqrt(mean_squared_error(true_real, pred_real))
            r2 = r2_score(true_real, pred_real)

            history[name]['RMSE'].append(rmse)
            history[name]['R2'].append(r2)

        if i == steps: break

        # 3. Active Selection (Global Uncertainty)
        X_pool = X[pool_idx]
        _, pool_vars = predict(model, likelihood, X_pool)

        best_local = torch.argmax(pool_vars).item()
        best_global = pool_idx[best_local]

        train_idx.append(best_global)
        pool_idx.pop(best_local)

        if i % 10 == 0:
            print(f"  Stage 2 Iter {i}/{steps}: Added sample. Train Size: {len(train_idx)}")

    return history

# ==========================================
# 5. EXECUTION
# ==========================================

PSP4ALL_FILE = 'psp4all.csv'
PCA_FILE = 'pca_pspall7.csv'

# 1. Load
X, y, indices, scaler_y = load_data(PSP4ALL_FILE, PCA_FILE)

# 2. Run Stage 1 (Train M0 on 4-comp)
# We start with 10 random samples and actively add 20 more 4-comp samples
# Result: A 30-sample training set optimized for 4-comp space
stage1_train_idx = run_stage_1_initialization(X, y, indices['4comp_total'], start_size=10, steps=50)

# 3. Run Stage 2 (Improve M0 via Global AL)
# We add 50 more samples from the "Global Pool"
metrics = run_stage_2_extrapolation(X, y, stage1_train_idx, indices, scaler_y, steps=200)

# ==========================================
# 6. VISUALIZATION
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Stage 2 Results: Active Improvement on Global Dataset\n(Started with {len(stage1_train_idx)} samples from Stage 1)", fontsize=14)

# Plot RMSE
ax = axes[0]
markers = ['o', 's', '^', 'D']
for j, (name, h) in enumerate(metrics.items()):
    ax.plot(h['RMSE'], label=name, marker=markers[j], markersize=4)
ax.set_title("RMSE Reduction")
ax.set_ylabel("RMSE (GPa)")
ax.set_xlabel("Samples Added (Stage 2)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot R2
ax = axes[1]
for j, (name, h) in enumerate(metrics.items()):
    ax.plot(h['R2'], label=name, marker=markers[j], markersize=4)
ax.set_title("R² Score Improvement")
ax.set_ylabel("R²")
ax.set_xlabel("Samples Added (Stage 2)")
ax.grid(True, alpha=0.3)

plt.show()

# Print Final Stats
print("\n--- Final Performance (End of Stage 2) ---")
for name, h in metrics.items():
    print(f"{name:<15} | Final RMSE: {h['RMSE'][-1]:.2f} GPa | Final R2: {h['R2'][-1]:.3f}")

# ============================================================
# REVISED MAIN SCRIPT: 4-Comp Training -> 7-Comp Prediction
# Strategy: Train on PCs->FormationEnergy.
# Validate using known Bulk Modulus manifold.
# ============================================================

# 1. Load Data
# We assume the file contains both 4-comp and 7-comp data
df = pd.read_csv('pca_pspall7.csv')

# 2. Separate Data based on availability of Formation Energy
# Train set = 4-component samples (Has Formation Energy & Bulk Modulus)
# Inference set = 7-component samples (Has Bulk Modulus, Missing Formation Energy)
# We assume missing energy is NaN or you can filter by element count if preferred.
# adjusting filter to ensure we catch the right rows:
if 'formation_energy' in df.columns:
    # Train on rows that HAVE formation energy
    train_mask = df['formation_energy'].notna() & (df['formation_energy'] != 0)
    # Predict on rows that DO NOT have formation energy (but have PCs)
    # OR if you want to force specific indices based on your knowledge of the dataset:
    target_mask = ~train_mask
else:
    raise ValueError("Column 'formation_energy' not found in dataset.")

df_train = df[train_mask].copy()
df_target = df[target_mask].copy()

print(f"Training Model on {len(df_train)} samples (4-comp)")
print(f"Predicting on {len(df_target)} samples (7-comp)")

# 3. Setup Features (PCs ONLY)
# We strictly use PCs as design inputs. Bulk Modulus is NOT used here.
n_pcs = 3
pc_cols = [c for c in df.columns if c.startswith("PC")][:n_pcs]

X_train_raw = df_train[pc_cols].values
y_train = df_train['formation_energy'].values

X_target_raw = df_target[pc_cols].values
# We keep Bulk Modulus separately for plotting/validation later
bm_train = df_train['bulk_modulus'].values
bm_target = df_target['bulk_modulus'].values

# 4. Global Scaling of Inputs
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
# Apply same scaling to target set
X_target = scaler_X.transform(X_target_raw)

# 5. Train the Model (Formation Energy)
print("\n[Training] Fitting GP on 4-component Formation Energy...")
# Re-using your existing class
gp_model = GPyTorchGPR_ARDSE(n_pcs=n_pcs, device="cuda" if torch.cuda.is_available() else "cpu", ardse='ardse')
gp_model.fit(X_train, y_train, n_epochs=300, lr=0.1)

# 6. Predict on 7-component samples
print("[Inference] Predicting 7-component Formation Energy...")
y_pred, y_std = gp_model.predict(X_target, return_std=True)

# Store results
df_target['pred_formation_energy'] = y_pred
df_target['pred_uncertainty'] = y_std

# ============================================================
# VISUALIZATION: The "Physical Consistency" Manifold
# ============================================================

def plot_manifold_validation(bm_train, y_train, bm_target, y_pred, y_std, fname="Figure_8_Manifold_Check.png"):
    """
    Plots Bulk Modulus vs Formation Energy.
    Since we have true Bulk Modulus for ALL samples, this plot reveals if
    the predicted Formation Energies obey physical laws (correlation with Bulk Modulus).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Plot 4-Comp Ground Truth (The "Reference Manifold")
    ax.scatter(bm_train, y_train, c='silver', s=50, alpha=0.4,
               label='4-Comp Ground Truth', edgecolors='none', zorder=1)

    # 2. Plot 7-Comp Predictions
    # Color by uncertainty: Darker = Confident, Lighter/Yellow = Uncertain
    sc = ax.scatter(bm_target, y_pred, c=y_std, cmap='viridis_r', s=60,
                    marker='o', edgecolors='k', linewidth=0.5, alpha=0.9,
                    label='7-Comp Predictions', zorder=2)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Prediction Uncertainty ($\sigma$)', fontweight='bold')

    # Formatting
    ax.set_xlabel('Bulk Modulus (GPa) [Known Ground Truth]', fontweight='bold', fontsize=12)
    ax.set_ylabel('Formation Energy (eV/atom) [Predicted]', fontweight='bold', fontsize=12)
    ax.set_title('Validation: Do predictions follow the physical trend?', fontweight='bold')

    # Add a trendline for the training data to guide the eye
    z = np.polyfit(bm_train, y_train, 2)
    p = np.poly1d(z)
    range_bm = np.linspace(bm_train.min(), bm_target.max(), 100)
    ax.plot(range_bm, p(range_bm), "k--", alpha=0.5, label='Trend (4-comp)')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"Saved {fname}")

plot_manifold_validation(bm_train, y_train, bm_target, y_pred, y_std)

# ============================================================
# EXPORT TOP CANDIDATES FOR DFT
# ============================================================
# Identify samples with Low Formation Energy (Stable) AND Low Uncertainty (Confident)
# Score = Energy + w * Sigma (Lower is better)
w = 1.0
df_target['acquisition_score'] = df_target['pred_formation_energy'] - w * df_target['pred_uncertainty']

# Select 10 random samples as requested for verification
random_samples = df_target#.sample(n=10, random_state=42)

print("\n--- Selected 10 Random Samples for DFT Verification ---")
cols_to_show = ['formula', 'bulk_modulus', 'pred_formation_energy', 'pred_uncertainty']
if 'formula' not in df_target.columns:
    cols_to_show[0] = pc_cols[0] # Fallback if formula missing

print(random_samples[cols_to_show])
random_samples[cols_to_show].to_csv('dft_verification_samples.csv', index=False)
print("Saved dft_verification_samples.csv")

# ============================================================
# REVISED MAIN SCRIPT: Transfer Learning (4-comp -> 7-comp)
# ============================================================

# 1. Load Data
df = pd.read_csv('pca_pspall7.csv')

# 2. Identify Training set (4-comp, has formation energy) vs Target set (7-comp, missing formation energy)
# We assume 'formation_energy' is NaN for the 7-comp samples you want to predict
# If your CSV has 0.0 or a dummy value for the 7-comp formation energy, adjust the filter below.
train_mask = df['formation_energy'].notna() & (df['formation_energy'] != 0)
target_mask = ~train_mask

df_train = df[train_mask].copy()  # 4-component samples
df_target = df[target_mask].copy() # 7-component samples

print(f"Training on {len(df_train)} samples (4-comp/Has Energy)")
print(f"Predicting on {len(df_target)} samples (7-comp/No Energy)")

# 3. Construct Features (X) using PCs AND Bulk Modulus
# By adding Bulk Modulus as a feature, we anchor the Formation Energy prediction to a known physical property.
n_pcs = 3
pc_cols = [c for c in df.columns if c.startswith("PC")][:n_pcs]

# Training X: PCs + Bulk Modulus
X_train_raw = np.hstack([
    df_train[pc_cols].values,
    df_train[['bulk_modulus']].values
])
y_train = df_train['formation_energy'].values

# Target X: PCs + Bulk Modulus (We have BM for these!)
X_target_raw = np.hstack([
    df_target[pc_cols].values,
    df_target[['bulk_modulus']].values
])

# 4. Global Scaling
# Important: Scale based on the Training data statistics, apply to Target
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_target = scaler_X.transform(X_target_raw)

# 5. Train GP Model (Using your existing GPyTorchGPR_ARDSE class)
# Note: n_pcs is now 4 (3 PCs + 1 Bulk Modulus)
gp_model = GPyTorchGPR_ARDSE(n_pcs=X_train.shape[1], device="cuda" if torch.cuda.is_available() else "cpu", ardse='ardse')

print("\nFitting GP on 4-component data...")
gp_model.fit(X_train, y_train, n_epochs=300, lr=0.1)

# 6. Predict on 7-component samples
print("Predicting 7-component formation energies...")
y_pred, y_std = gp_model.predict(X_target, return_std=True)

# Store results in the dataframe
df_target['predicted_formation_energy'] = y_pred
df_target['prediction_uncertainty'] = y_std

# ============================================================
# VISUALIZATION: Property-Property Manifold
# ============================================================

def plot_transfer_results(df_train, df_target, fname="Figure_8_Transfer_Prediction.png"):
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Plot Training Data (Ground Truth)
    # This establishes the physical "Trend" between Bulk Modulus and Formation Energy
    sc1 = ax.scatter(df_train['bulk_modulus'], df_train['formation_energy'],
                     c='gray', alpha=0.5, label='4-Comp Ground Truth', s=40, edgecolors='k')

    # 2. Plot Target Predictions (7-Comp)
    # We color these by their Uncertainty (Sigma)
    # If the predictions are good, they should generally align with the gray dots' trend
    sc2 = ax.scatter(df_target['bulk_modulus'], df_target['predicted_formation_energy'],
                     c=df_target['prediction_uncertainty'], cmap='plasma_r',
                     label='7-Comp Predictions', s=80, marker='*', edgecolors='k', zorder=10)

    cbar = plt.colorbar(sc2, ax=ax)
    cbar.set_label('Prediction Uncertainty ($sigma$)', fontweight='bold')

    ax.set_xlabel('Bulk Modulus (GPa) [Known Input]', fontweight='bold', fontsize=12)
    ax.set_ylabel('Formation Energy (eV/atom) [Predicted]', fontweight='bold', fontsize=12)
    ax.set_title('Inference Validation: Consistency with Physical Manifold', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"Saved {fname}")

plot_transfer_results(df_train, df_target)

# ============================================================
# SELECTION FOR DFT VALIDATION
# ============================================================
# Instead of random 10 samples, pick 5 High Confidence (Low Sigma) and 5 Exploratory (High Sigma)
# or just random as you requested.

# Method A: Random 10 (as requested)
random_indices = np.random.choice(df_target.index, 10, replace=False)
samples_to_validate = df_target#.loc[random_indices]

# Method B (Optional Smart Selection):
# samples_to_validate = df_target.nlargest(10, 'prediction_uncertainty')

print("\nSamples selected for DFT Verification:")
out_cols = ['formula', 'bulk_modulus', 'predicted_formation_energy', 'prediction_uncertainty']
# Check if 'formula' exists, otherwise use index
if 'formula' not in df.columns:
    out_cols[0] = pc_cols[0]

print(samples_to_validate[out_cols])
samples_to_validate[out_cols].to_csv("dft_validation_candidates.csv")

sc
tackle metabdimport os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, LCMKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# ============================================================
# Configuration
# ============================================================


TRAIN_ITERS = 500  # Increased for better convergence
N_PCS = 3  # Number of principal components to use
RANDOM_STATE = 42
SAVED = './'  # Output directory

print(f"Using device: {DEVICE}")


# ============================================================
# Error Metrics
# ============================================================

class ErrorMetrics:
    """Regression error metrics for single and multi-output predictions."""

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred, multioutput='raw_values')

    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, multioutput='raw_values')

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(ErrorMetrics.mse(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        """Safe MAPE with epsilon for near-zero values."""
        epsilon = 1e-6 * np.max(np.abs(y_true), axis=0)
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)), axis=0) * 100

    @staticmethod
    def nmae(y_true, y_pred):
        """Normalized MAE as percentage of range."""
        mae = ErrorMetrics.mae(y_true, y_pred)
        ranges = np.max(y_true, axis=0) - np.min(y_true, axis=0)
        return mae / (ranges + 1e-10) * 100

    @staticmethod
    def r_squared(y_true, y_pred):
        """R² for each output."""
        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
        r2 = np.zeros(n_outputs)
        for i in range(n_outputs):
            r2[i] = r2_score(y_true[:, i], y_pred[:, i])
        return r2

    @staticmethod
    def compute_all(y_true, y_pred):
        """Compute all metrics for multi-output regression."""
        return {
            "MAE": ErrorMetrics.mae(y_true, y_pred),
            "MSE": ErrorMetrics.mse(y_true, y_pred),
            "RMSE": ErrorMetrics.rmse(y_true, y_pred),
            "MAPE": ErrorMetrics.mape(y_true, y_pred),
            "NMAE": ErrorMetrics.nmae(y_true, y_pred),
            "R2": ErrorMetrics.r_squared(y_true, y_pred),
        }


# ============================================================
# Multitask GP Model
# ============================================================

class MultitaskGPModel(ExactGP):
    """
    Multitask Gaussian Process with Linear Coregionalization Model (LCM) kernel.

    Predicts multiple correlated outputs (bulk modulus + formation energy) jointly,
    learning correlations between tasks for improved predictions.
    """

    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        n_dims = train_x.shape[1]

        # Multitask mean (shared constant mean per task)
        self.mean_module = MultitaskMean(
            ConstantMean(),
            num_tasks=num_tasks
        )

        # LCM kernel: learns correlations between tasks
        # Uses two latent RBF kernels with ARD (one per task)
        self.covar_module = LCMKernel(
            base_kernels=[
                RBFKernel(ard_num_dims=n_dims),
                RBFKernel(ard_num_dims=n_dims)
            ],
            num_tasks=num_tasks,
            rank=num_tasks  # Full rank for maximum flexibility
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


# ============================================================
# GP Wrapper Class
# ============================================================

class MultitaskGPR:
    """
    Wrapper for Multitask Gaussian Process Regression.

    Handles:
    - Input/output scaling
    - Model training with MLL optimization
    - Prediction with uncertainty quantification
    """

    def __init__(self, n_features, num_tasks=2, device="cpu"):
        self.n_features = n_features
        self.num_tasks = num_tasks
        self.device = DEVICE

        self.model = None
        self.likelihood = None
        self._fitted = False

        # Scalers for inputs and outputs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Store training data
        self.train_X = None
        self.train_Y = None

    def fit(self, X, y, n_epochs=200, lr=0.1, verbose=True):
        """
        Fit the multitask GP model.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, num_tasks)
            n_epochs: Number of optimization iterations
            lr: Learning rate for Adam optimizer
            verbose: Print training progress
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Fit and transform scalers
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Convert to tensors
        self.train_X = torch.FloatTensor(X_scaled).to(self.device)
        self.train_Y = torch.FloatTensor(y_scaled).to(self.device)

        # Initialize model and likelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_tasks).to(self.device)
        self.model = MultitaskGPModel(
            self.train_X,
            self.train_Y,
            self.likelihood,
            num_tasks=self.num_tasks
        ).to(self.device)

        # Training mode
        self.model.train()
        self.likelihood.train()

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training loop
        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -mll(output, self.train_Y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}: Loss = {loss.item():.4f}")

        self._fitted = True
        return losses

    def predict(self, X, return_std=True):
        """
        Make predictions with uncertainty quantification.

        Returns predictions in ORIGINAL units (inverse-transformed).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            preds = self.likelihood(self.model(X_tensor))
            y_mean_scaled = preds.mean.cpu().numpy()
            y_var_scaled = preds.variance.cpu().numpy()

        # Inverse transform to original units
        y_mean = self.scaler_y.inverse_transform(y_mean_scaled)

        # Scale standard deviation back to original units
        y_std = np.sqrt(y_var_scaled) * self.scaler_y.scale_

        if return_std:
            return y_mean, y_std
        return y_mean

    def count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(X, y, test_size=0.2, n_epochs=200, random_state=42):
    """
    Train multitask GPR and evaluate on train/test split.

    Args:
        X: Input features (PCA scores)
        y: Target values (bulk_modulus, formation_energy)
        test_size: Fraction for test set
        n_epochs: Training iterations
        random_state: Random seed for reproducibility

    Returns:
        model: Trained MultitaskGPR model
        metrics: Dictionary of train/test metrics
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Input features: {X.shape[1]}")
    print(f"Output tasks: {y.shape[1]}")
    print()

    # Initialize and train model
    model = MultitaskGPR(
        n_features=X.shape[1],
        num_tasks=y.shape[1],
        device=DEVICE
    )

    print("Training Multitask GPR...")
    losses = model.fit(X_train, y_train, n_epochs=n_epochs, lr=0.1, verbose=True)
    print(f"\nTrainable parameters: {model.count_parameters()}")

    # Predictions
    y_train_pred, y_train_std = model.predict(X_train, return_std=True)
    y_test_pred, y_test_std = model.predict(X_test, return_std=True)

    # Compute metrics
    train_metrics = ErrorMetrics.compute_all(y_train, y_train_pred)
    test_metrics = ErrorMetrics.compute_all(y_test, y_test_pred)

    # Store results
    results = {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_train_pred": y_train_pred, "y_test_pred": y_test_pred,
        "y_train_std": y_train_std, "y_test_std": y_test_std,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "losses": losses
    }

    return results


def print_metrics(results, property_names=["Bulk Modulus (GPa)", "Formation Energy (eV)"]):
    """Print formatted metrics for all properties."""
    train_metrics = results["train_metrics"]
    test_metrics = results["test_metrics"]

    for i, prop_name in enumerate(property_names):
        print("=" * 60)
        print(f"{prop_name}")
        print("=" * 60)
        print()
        print("Training Set:")
        print(f"  MAE:  {train_metrics['MAE'][i]:.4f}")
        print(f"  RMSE: {train_metrics['RMSE'][i]:.4f}")
        print(f"  MAPE: {train_metrics['MAPE'][i]:.2f}%")
        print(f"  NMAE: {train_metrics['NMAE'][i]:.2f}%")
        print(f"  R²:   {train_metrics['R2'][i]:.4f}")
        print()
        print("Test Set:")
        print(f"  MAE:  {test_metrics['MAE'][i]:.4f}")
        print(f"  RMSE: {test_metrics['RMSE'][i]:.4f}")
        print(f"  MAPE: {test_metrics['MAPE'][i]:.2f}%")
        print(f"  NMAE: {test_metrics['NMAE'][i]:.2f}%")
        print(f"  R²:   {test_metrics['R2'][i]:.4f}")
        print()


# ============================================================
# Plotting Functions
# ============================================================

def plot_predictions(results, property_names=["Bulk Modulus (GPa)", "Formation Energy (eV)"],
                    fname="GPR_predictions.png", color=None, rng=None):
    """Plot predictions vs ground truth for both properties."""

    y_train = results["y_train"]
    y_test = results["y_test"]
    y_train_pred = results["y_train_pred"]
    y_test_pred = results["y_test_pred"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Multitask GPR: Predictions vs Ground Truth", fontweight="bold", fontsize=14)

    for i, (ax, prop_name) in enumerate(zip(axes, property_names)):
        # Combine for axis limits
        all_true = np.concatenate([y_train[:, i], y_test[:, i]])
        all_pred = np.concatenate([y_train_pred[:, i], y_test_pred[:, i]])

        vmin = min(all_true.min(), all_pred.min())
        vmax = max(all_true.max(), all_pred.max())
        margin = (vmax - vmin) * 0.05


        if i==0:
            ax.plot([70, 170], [70, 170],'k--', linewidth=2)
        elif i==1:
            ax.plot([-55, 5], [-55, 5],'k--', linewidth=2)
        else:
            # Parity line
            ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],'k--', linewidth=2)

        # Scatter plots
        ax.scatter(y_train[:, i], y_train_pred[:, i],
                   c='#2E86AB', alpha=0.7, s=50, edgecolor='k', label='Train')
        if color ==None:
            ax.scatter(y_test[:, i], y_test_pred[:, i],
                      c='#A23B72', alpha=0.7, s=50, marker='s', edgecolor='k', label='Test')
        else:
            ax.scatter(y_test[:, i], y_test_pred[:, i],
                      c=color, alpha=0.7, s=50, marker='s', edgecolor='k', label='Test')

        # Labels and formatting
        ax.set_xlabel(f"DFT-Computed {prop_name}", fontweight="bold")
        ax.set_ylabel(f"ML-Predicted {prop_name}", fontweight="bold")
        if i==0:
            ax.set_xlim(70, 170)
            ax.set_ylim(70, 170)
        elif i==1:
            ax.set_xlim(-55, 5)
            ax.set_ylim(-55, 5)
        else:
            ax.set_xlim(vmin - margin, vmax + margin)
            ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_aspect('equal')
        #ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Add R² annotation
        test_r2 = results["test_metrics"]["R2"][i]
        ax.text(0.95, 0.05, f'Test R² = {test_r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")


def plot_residuals(results, property_names=["Bulk Modulus (GPa)", "Formation Energy (eV)"],
                   fname="GPR_residuals.png"):
    """Plot residuals for both properties."""

    y_test = results["y_test"]
    y_test_pred = results["y_test_pred"]
    y_test_std = results["y_test_std"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Multitask GPR: Residual Analysis (Test Set)", fontweight="bold", fontsize=14)

    for i, (ax, prop_name) in enumerate(zip(axes, property_names)):
        residuals = y_test[:, i] - y_test_pred[:, i]

        ax.scatter(y_test_pred[:, i], residuals,
                   c='#A23B72', alpha=0.7, s=50, edgecolor='k')
        ax.axhline(0, color='r', linestyle='--', linewidth=2)

        # Uncertainty band (±2σ mean)
        mean_std = np.mean(y_test_std[:, i])
        ax.axhspan(-2*mean_std, 2*mean_std, color='gray', alpha=0.2, label=f'±2σ (mean σ={mean_std:.3f})')

        ax.set_xlabel(f"Predicted {prop_name}", fontweight="bold")
        ax.set_ylabel("Residual (True - Predicted)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {fname}")


def plot_training_loss(results, fname="GPR_training_loss.png"):
    """Plot training loss curve."""

    losses = results["losses"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Negative Log Marginal Likelihood", fontweight="bold")
    ax.set_title("Multitask GPR Training Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"Saved {fname}")


# ============================================================
# Main Script
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Load Data
    # ----------------------------------------------------------

    # Load PCA scores dataset
    # Expected columns: PC1, PC2, ..., PCn, bulk_modulus, formation_energy

    df = pd.read_csv('pca_pspall.csv')

    # Check for required columns
    required_cols = ["bulk_modulus", "formation_energy"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Get PC columns
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if len(pc_cols) < N_PCS:
        raise ValueError(f"Need at least {N_PCS} PC columns in the CSV.")

    print(f"Dataset: {len(df)} samples")
    print(f"Using {N_PCS} principal components: {pc_cols[:N_PCS]}")
    print()

    # Extract features and targets
    X = df[pc_cols[:N_PCS]].values

    # Targets: Bulk Modulus and Formation Energy
    y_bm = df["bulk_modulus"].values
    y_fe = df["formation_energy"].values # Scale formation energy

    # Combine into multi-output target
    y = np.column_stack([y_bm, y_fe])

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    print(f"Bulk modulus range: {y_bm.min():.2f} - {y_bm.max():.2f} GPa")
    print(f"Formation energy range: {y_fe.min():.4f} - {y_fe.max():.4f} eV")
    print()

    # ----------------------------------------------------------
    # Train and Evaluate
    # ----------------------------------------------------------

    results = train_and_evaluate(
        X, y,
        test_size=0.99,
        n_epochs=TRAIN_ITERS,
        random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------
    # Print Metrics
    # ----------------------------------------------------------

    print_metrics(results, property_names=["Bulk Modulus (GPa)", "Bulk Modulus (GPa)"])

    # ----------------------------------------------------------
    # Generate Plots
    # ----------------------------------------------------------

    plot_predictions(results, fname=os.path.join(SAVED, "GPR_predictions.png"))
    plot_residuals(results, fname=os.path.join(SAVED, "GPR_residuals.png"))
    plot_training_loss(results, fname=os.path.join(SAVED, "GPR_training_loss.png"))

    # ----------------------------------------------------------
    # Save Metrics to CSV
    # ----------------------------------------------------------

    metrics_rows = []
    property_names = ["Bulk Modulus", "Formation Energy"]

    for i, prop in enumerate(property_names):
        metrics_rows.append({
            "Property": prop,
            "Set": "Train",
            "MAE": results["train_metrics"]["MAE"][i],
            "RMSE": results["train_metrics"]["RMSE"][i],
            "MAPE": results["train_metrics"]["MAPE"][i],
            "NMAE": results["train_metrics"]["NMAE"][i],
            "R2": results["train_metrics"]["R2"][i],
        })
        metrics_rows.append({
            "Property": prop,
            "Set": "Test",
            "MAE": results["test_metrics"]["MAE"][i],
            "RMSE": results["test_metrics"]["RMSE"][i],
            "MAPE": results["test_metrics"]["MAPE"][i],
            "NMAE": results["test_metrics"]["NMAE"][i],
            "R2": results["test_metrics"]["R2"][i],
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(SAVED, "GPR_metrics.csv"), index=False)
    print("\nSaved GPR_metrics.csv:")
    print(metrics_df.to_string(index=False))

    # ----------------------------------------------------------
    # Example: Make predictions on new data
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("Example: Predicting on first 5 samples")
    print("=" * 60)

    model = results["model"]
    X_sample = X[:5]
    y_sample = y[:5]

    y_pred, y_std = model.predict(X_sample, return_std=True)

    print("\nSample predictions:")
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  Bulk Modulus:     True={y_sample[i,0]:.2f}, Pred={y_pred[i,0]:.2f} ± {y_std[i,0]:.2f} GPa")
        print(f"  Formation Energy: True={y_sample[i,1]:.4f}, Pred={y_pred[i,1]:.4f} ± {y_std[i,1]:.4f} eV")

