"""
Auto-generated module from Jupyter Notebook
"""

import sys
import numpy as np
import math
import h5py
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

DIR='/content/drive'
HOME = os.path.join(DIR, 'MyDrive')
HOMED = os.path.join(HOME, "Analyze4RHEA")#Analyze7RHEA
os.chdir(HOMED)

from typing import Union, List, Optional
from itertools import combinations
from collections import Counter
from typing import List, Tuple


import re
from itertools import permutations

def generate_formula_combinations(formula: str):
    """
    Given a formula like 'Al8Nb92Ti20Zr8', generate all
    distinct permutations of the element-count blocks,
    e.g. ['Al8Nb92Ti20Zr8', 'Al8Nb92Zr8Ti20', ...].
    """
    # Split into element+count tokens: ['Al8', 'Nb92', 'Ti20', 'Zr8']
    tokens = re.findall(r'[A-Z][a-z]?\d*', formula)

    # Generate all unique permutations of these tokens
    perm_strings = {''.join(p) for p in permutations(tokens)}

    # Return as a sorted list (optional)
    return sorted(perm_strings)

class AlloyLookup:
    """Class to manage alloy composition lookups by formula."""

    def __init__(self, csv_file='alloy_atoms.csv'):
        """
        Initialize the lookup system.

        Parameters:
        -----------
        csv_file : str
            Path to the alloy_atoms.csv file
        """
        try:
            self.df = pd.read_csv(csv_file)
            self.total_rows = len(self.df)
            print(f"✓ Loaded {self.total_rows} alloy compositions from '{csv_file}'")
        except FileNotFoundError:
            print(f"❌ Error: File '{csv_file}' not found.")
            raise

    def lookup_single(self, formula: str) -> Optional[pd.Series]:
        """
        Look up a single formula and return the complete row.

        Parameters:
        -----------
        formula : str
            Chemical formula (e.g., 'Al4Nb4Ti4Zr116')

        Returns:
        --------
        pd.Series or None
            Row data if found, None if not found
        """
        result = None
        #print(generate_formula_combinations(formula))
        for f in generate_formula_combinations(formula):
            print(f)
            temp = self.df[self.df['Formula'] == f]
            #print(f"temp: {temp}")
            #print(f"temp.empty: {temp.empty}")

            # Check if the dataframe is not empty (meaning a match was found)
            if not temp.empty:
                result = temp
                break

        if len(result) == 0:
            print(f"❌ Formula '{formula}' not found in dataset.")
            return None
        elif len(result) == 1:
            print(f"✓ Found formula: {formula}")
            return result.iloc[0]
        else:
            print(f"⚠ Warning: Multiple matches found for '{formula}' ({len(result)} entries)")
            return result

    def lookup_multiple(self, formula_list: List[str]) -> Optional[pd.DataFrame]:
        """
        Look up multiple formulas and return results as a DataFrame.

        Parameters:
        -----------
        formula_list : List[str]
            List of chemical formulas to search for

        Returns:
        --------
        pd.DataFrame or None
            DataFrame with all matching rows, None if no matches found
        """
        results = self.df[self.df['Formula'].isin(formula_list)]

        if len(results) == 0:
            print("❌ No matching formulas found.")
            return None

        print(f"✓ Found {len(results)} out of {len(formula_list)} formulas")

        # Report missing formulas
        if len(results) < len(formula_list):
            found_formulas = set(results['Formula'].tolist())
            missing_formulas = set(formula_list) - found_formulas
            print(f"⚠ Missing formulas ({len(missing_formulas)}):")
            for formula in sorted(missing_formulas):
                print(f"   - {formula}")

        return results

    def get_properties(self, formula: str) -> dict:
        """
        Get just the computed properties for a formula.

        Parameters:
        -----------
        formula : str
            Chemical formula

        Returns:
        --------
        dict
            Dictionary with Formation_Energy, Bulk_Modulus, and composition
        """
        result = self.lookup_single(formula)
        if result is None:
            return {}

        return {
            'Formation_Energy': result['Formation_Energy'],
            'Bulk_Modulus': result['Bulk_Modulus'],
            'Al': int(result['Al']),
            'Nb': int(result['Nb']),
            'Ti': int(result['Ti']),
            'Zr': int(result['Zr']),
        }

    def print_row(self, formula: str, columns: Optional[List[str]] = None) -> None:
        """
        Print a formatted row for a single formula.

        Parameters:
        -----------
        formula : str
            Chemical formula
        columns : List[str], optional
            Columns to display. If None, displays all columns
        """
        result = self.lookup_single(formula)
        if result is None:
            return

        if columns is None:
            columns = result.index.tolist()

        print("\n" + "=" * 60)
        print(f"Formula: {formula}")
        print("=" * 60)
        for col in columns:
            if col in result.index:
                value = result[col]
                # Format floats nicely
                if isinstance(value, float):
                    print(f"  {col:20s}: {value:.6g}")
                else:
                    print(f"  {col:20s}: {value}")
        print("=" * 60 + "\n")

    def search_by_composition(self, al: Optional[int] = None, nb: Optional[int] = None,
                             ti: Optional[int] = None, zr: Optional[int] = None) -> pd.DataFrame:
        """
        Search for alloys by atomic composition.

        Parameters:
        -----------
        al, nb, ti, zr : int, optional
            Number of atoms for each element

        Returns:
        --------
        pd.DataFrame
            All matching rows
        """
        result = self.df.copy()

        if al is not None:
            result = result[result['Al'] == al]
        if nb is not None:
            result = result[result['Nb'] == nb]
        if ti is not None:
            result = result[result['Ti'] == ti]
        if zr is not None:
            result = result[result['Zr'] == zr]

        if len(result) == 0:
            print("❌ No matching compositions found.")
            return None

        print(f"✓ Found {len(result)} matching compositions")
        return result


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


def plot_figure5_pc_space(df, pc_cols, target_col="bulk_modulus", target_col2="formation_energy",
                          fname="Figure_5_PC_Space.png"):
    if len(pc_cols) < 3:
        print("Need at least 3 PC columns for Figure 5.")
        return

    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    target = df[target_col].values
    target2 = df[target_col2].values

    fig = plt.figure(figsize=(12, 5))

    # Bulk modulus plot
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(pc1, pc2, c=target, cmap="viridis",
                      s=40, edgecolor="k", alpha=0.8)
    ax1.set_xlabel(pc_cols[0], fontweight="bold")
    ax1.set_ylabel(pc_cols[1], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target2, cmap="viridis",
                      s=40, edgecolor="k", alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Alloy Formation Energy (eV)', fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

rhos = []
formulas = []
c=0
forms =[]
bulks =[]

for fname in os.listdir('./SQS2PS/'): #SQS2PS #Relax2PS
    try:
      print(fname)
    except:
      h5f.close()
    save_path= os.path.join(HOMED,'SQS2PS',fname)
    with h5py.File(save_path, 'r') as f:
        rho3d = f['tpt'][:]
        c+=1
        formula = fname[:-3]
        if formula[-1]==')':
          print(f"Skipped: {formula}")
          continue
        lookup = AlloyLookup('RHEA4unique.csv') # alloy_atoms  RHEA4unique
        row = lookup.lookup_single(formula)
        formulas.append(formula)
        print(row)
        forms.append(row['formation_energy'])
        bulks.append(row['Bulk'])
        print(formula, c)
        rhos.append(rho3d[:].flatten())
        print(np.shape(rho3d))
        f.close()

for fname in os.listdir('./SQS2PS_rest/'): #SQS2PS #Relax2PS
    try:
      print(fname)
    except:
      h5f.close()
    save_path= os.path.join(HOMED,'SQS2PS_rest',fname)
    with h5py.File(save_path, 'r') as f:
        rho3d = f['tpt'][:]
        c+=1
        formula = fname[:-3]
        if formula[-1]==')':
          print(f"Skipped: {formula}")
          continue
        lookup = AlloyLookup('RHEA4unique.csv')
        row = lookup.lookup_single(formula)
        formulas.append(formula)
        print(row)
        forms.append(row['formation_energy'])
        bulks.append(row['Bulk'])
        print(formula, c)
        rhos.append(rho3d[:].flatten())
        print(np.shape(rho3d))
        f.close()#"""

print(np.shape(rhos))

# integral_rho = np.dot(S['W'], S['rho_at'])
# print(f'Integral rho before scaling: {integral_rho}')

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rhos = scaler.fit_transform(rhos)

# Assuming 'rhos' is your input numpy array.
# Replace this with your actual data loading mechanism if needed.
# Example data generation for demonstration:
# For a real scenario, your 'rhos' array should have dimensions (n_samples, n_features)
# and n_features must be at least 30.

# 1. Get PCA (30 components)
n_components = 50
# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

principal_components = pca.fit_transform(np.asarray(rhos))

# 3. Create a pandas DataFrame from the principal components
# Assign meaningful column names (e.g., PC1, PC2, ...).
pc_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(data=principal_components, columns=pc_columns)

# 4. Add the 'formulas' array as an identifier column
# This assumes 'formulas' has the same number of rows as 'rhos' (or df_pca).
df_pca['formula'] = formulas
#df_pca['formation_energy'] = forms
df_pca['bulk_modulus'] = bulks

# Optional: You can also add the explained variance ratio to understand how much
# variance each component explains.
# print("Explained variance ratio per component:", pca.explained_variance_ratio_)
# print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

# 5. Save the final DataFrame to a CSV file
df_pca.to_csv('pca_pspall7.csv', index=False) #pca_psp #pca_psp2

print(f"PCA results")

df = df_pca.copy()

# Check if we need to find the REAL data file
if len(df) < 1000:  # If it's just a small example
    print("Looking for full dataset with 4495 samples...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        print(f"Found full dataset: {csv_path} ({os.path.getsize(csv_path)} bytes)")
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)

# Verify columns exist
required_cols = ["bulk_modulus", "formation_energy"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    if df[col].isna().any():
        raise ValueError(f"Column {col} has NaN values. Please provide complete dataset.")

# PC columns
pc_cols = [c for c in df.columns if c.startswith("PC")]
if len(pc_cols) < 3:
    raise ValueError("Need at least PC1, PC2, PC3 in the CSV.")

# Use first 3 PCs as features
n_pcs = 3
X = df[pc_cols[:n_pcs]].values
y_bm = df["bulk_modulus"].values
y_fe = df["formation_energy"].values

# Get formula column for elemental initialization
formula_col = None
for col in ["formula", "composition", "comp", "Formula"]:
    if col in df.columns:
        formula_col = col
        break

if formula_col is None:
    print("Warning: No formula column found. Will use random initialization.")
    formulas = np.array([f"Sample{i}" for i in range(len(df))])
else:
    formulas = df[formula_col].values

print(f"✓ Loaded full dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")
print(f"Bulk modulus range: {y_bm.min():.2f} - {y_bm.max():.2f} GPa")
print(f"Formation energy range: {y_fe.min():.2f} - {y_fe.max():.2f} eV/atom")

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# --- Figure 3: PCA variance explained ---
plot_figure3_pca_variance(df, pc_cols)

# --- Figure 5: PC space visualization ---
if len(df) >= 100:
    plot_figure5_pc_space(df, pc_cols, target_col="bulk_modulus")

#df_pca.to_csv('pca_pspall.csv', index=False) #pca_psp #pca_psp2
#
#print(f"PCA results")

df = pd.read_csv('pca_pspall.csv')
# PC columns
pc_cols = [c for c in df.columns if c.startswith("PC")]
if len(pc_cols) < 3:
    raise ValueError("Need at least PC1, PC2, PC3 in the CSV.")

def pc3(df, pc_cols, target_col="bulk_modulus", target_col2="formation_energy",
                          fname="PC123.png"):
    if len(pc_cols) < 3:
        print("Need at least 3 PC columns for Figure 5.")
        return

    pc1 = df[pc_cols[0]].values
    pc2 = df[pc_cols[1]].values
    pc3 = df[pc_cols[2]].values
    target = df[target_col].values
    target2 = df[target_col2].values

    fig = plt.figure(figsize=(14, 15))

    # Bulk modulus plot
    ax1 = fig.add_subplot(3, 2, 1)
    sc1 = ax1.scatter(pc1, pc2, c=target, cmap="viridis",
                      s=40, alpha=0.8)
    ax1.set_xlabel(pc_cols[0], fontweight="bold")
    ax1.set_ylabel(pc_cols[1], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax2 = fig.add_subplot(3, 2, 2)
    sc2 = ax2.scatter(pc1, pc2, c=target2, cmap="rainbow",
                      s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Alloy Formation Energy (eV)', fontsize=9)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    # Bulk modulus plot
    ax3 = fig.add_subplot(3, 2, 3)
    sc1 = ax3.scatter(pc1, pc3, c=target, cmap="viridis",
                      s=40, alpha=0.8)
    ax3.set_xlabel(pc_cols[0], fontweight="bold")
    ax3.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax3, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax4 = fig.add_subplot(3, 2, 4)
    sc2 = ax4.scatter(pc1, pc3, c=target2, cmap="rainbow",
                      s=40, alpha=0.8)
    ax4.set_xlabel(pc_cols[0], fontweight="bold")
    ax4.set_ylabel(pc_cols[2], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax4)
    cbar2.set_label('Alloy Formation Energy (eV)', fontsize=9)

    ax3.set_box_aspect(1)
    ax4.set_box_aspect(1)

    # Bulk modulus plot
    ax5 = fig.add_subplot(3, 2, 5)
    sc1 = ax5.scatter(pc2, pc3, c=target, cmap="viridis",
                      s=40, alpha=0.8)
    ax5.set_xlabel(pc_cols[1], fontweight="bold")
    ax5.set_ylabel(pc_cols[2], fontweight="bold")
    cbar1 = fig.colorbar(sc1, ax=ax5, pad=0.1)
    cbar1.set_label('Bulk Modulus (GPa)', fontsize=9)

    # Formation energy plot
    ax6 = fig.add_subplot(3, 2, 6)
    sc2 = ax6.scatter(pc2, pc3, c=target2, cmap="rainbow",
                      s=40, alpha=0.8)
    ax6.set_xlabel(pc_cols[1], fontweight="bold")
    ax6.set_ylabel(pc_cols[2], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax6)
    cbar2.set_label('Alloy Formation Energy (eV)', fontsize=9)

    ax5.set_box_aspect(1)
    ax6.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname}")

    from mpl_toolkits.mplot3d import Axes3D

    # Create a figure and a 3D subplot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pnt3d = ax.scatter(pc1, pc2, pc3, c=target, cmap='viridis', marker='o')

    # 4. Add labels and a color bar
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    #ax.set_title('3D PC Plot Colored by Values')

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(pnt3d, ax=ax, pad=0.1)
    cbar.set_label('Bulk Modulus (GPa)')

    # Add a legend
    #ax.legend()

    # Save the plot to a file
    # You can change the filename and format as needed (e.g., .png, .jpg, .svg, .pdf)
    plt.savefig('pc123_bulk.png', dpi=300)

    # Display the plot (optional)
    plt.show()

    # Create a figure and a 3D subplot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    pnt3d = ax.scatter(pc1, pc2, pc3, c=target2, cmap='rainbow', marker='o')

    # 4. Add labels and a color bar
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    #ax.set_title('3D PC Plot Colored by Values')

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(pnt3d, ax=ax, pad=0.1)
    cbar.set_label('Alloy Formation Energy (eV)')

    # Add a legend
    #ax.legend()

    # Save the plot to a file
    # You can change the filename and format as needed (e.g., .png, .jpg, .svg, .pdf)
    plt.savefig('pc123_form.png', dpi=300)

    # Display the plot (optional)
    plt.show()

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
    target2 = df[target_col2].values

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
    sc2 = ax2.scatter(pc1, pc2, c=target2, cmap="rainbow",
                      s=40, alpha=0.8)
    ax2.set_xlabel(pc_cols[0], fontweight="bold")
    ax2.set_ylabel(pc_cols[1], fontweight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label('Alloy Formation Energy (eV)', fontsize=9)

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
                  ax2.scatter(subset['PC1'], subset['PC2'], edgecolors='black', label = 'Pure element', s=100, marker=marker, facecolors='None')
                else:
                  ax2.scatter(subset['PC1'], subset['PC2'], edgecolors='black', s=100, marker=marker, facecolors='None')
                ax2.legend()

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
                        label=f"{label}>{n}%",      # Legend label
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

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #pnt3d = ax.scatter(pc1, pc2, pc3, c=target2, cmap='rainbow', marker='o')
    pnt3d = ax.scatter(df['PC1'], df['PC2'], df['PC3'],
            c='lightgray', label='All Data', s=100, alpha=0.1, zorder=1)

    # 4. Add labels and a color bar
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    #ax.set_title('3D PC Plot Colored by Values')
    for label, mask, color, marker in layers:
        subset = df[mask]

        if not subset.empty:
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'],
                        c=color,          # Color defined above
                        label=f"{label}>{n}%",      # Legend label
                        marker=marker,    # Different markers help distinguish overlaps
                        s=100,             # Slightly larger size for highlights
                        alpha=0.8,        # Slight transparency
                        edgecolors='k',   # Black edge to make points pop
                        linewidth=0.5,
                        zorder=2)

    # Add a color bar which maps values to colors
    #cbar = fig.colorbar(pnt3d, ax=ax, pad=0.1)
    #cbar.set_label('Alloy Formation Energy (eV)')

    # Add a legend
    ax.legend()

    # Save the plot to a file
    # You can change the filename and format as needed (e.g., .png, .jpg, .svg, .pdf)
    plt.savefig('pc123SQS.png', dpi=300)

    # Display the plot (optional)
    plt.show()

pc3(df, pc_cols, target_col="bulk_modulus")
df_final = process_alloy_data(df)
pcquad(df_final, pc_cols, n=75,target_col="bulk_modulus", fname='pcquad4only.png')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRECTED VASt framework - matches paper exactly

Key fixes:
1. FIXED TEST SET (20% holdout, never used for selection)
2. INFORMATION GAIN: I(x) = σ(x) (absolute uncertainty)
3. INITIALIZATION: Elemental compositions (Al, Nb, Ti, Zr)
4. SEPARATE MODELS: Different GPR for each property
5. ARDSE kernel: length_scale = [n_pcs, n_pcs, n_pcs]
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C, Matern, WhiteKernel
)
from sklearn.model_selection import train_test_split

# ============================================================
# Error metrics
# ============================================================

class ErrorMetrics:
    """All ML error metrics used in the paper."""

    @staticmethod
    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = np.abs(y_true) > 1e-10
        if not np.any(mask):
            return 0.0
        return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    @staticmethod
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def nmae(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rng = np.max(y_true) - np.min(y_true)
        if rng <= 0:
            return 0.0
        return float(100.0 * np.mean(np.abs(y_true - y_pred)) / rng)

    @staticmethod
    def r_squared(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot <= 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

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
# Sklearn Gaussian Process with ARDSE kernel
# ============================================================

class SklearnGPR_ARDSE:
    """Gaussian Process Regression with ARDSE kernel (RBF with individual length scales)."""

    def __init__(self, n_pcs, alpha=1e-6):
        # ARDSE: length_scale is a vector, one per feature
        # Set all length scales equal to n_pcs (as requested)
        length_scale = np.full(n_pcs, fill_value=float(n_pcs))

        # ARDSE kernel: RBF with individual length scales per dimension
        ard_kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))

        kernel = C(1.0, (1e-3, 1e3)) * ard_kernel + WhiteKernel(noise_level=alpha)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-10,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b"
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._fitted = False
        self.n_pcs = n_pcs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y).ravel()
        self.model.fit(Xs, ys)
        self._fitted = True

    def predict(self, X, return_std=True):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = np.asarray(X)
        Xs = self.scaler_X.transform(X)
        y_mean_s, y_std_s = self.model.predict(Xs, return_std=True)
        y_mean = self.scaler_y.inverse_transform(y_mean_s.reshape(-1, 1)).ravel()
        y_std = y_std_s * self.scaler_y.scale_[0]
        if return_std:
            return y_mean, y_std
        return y_mean


# ============================================================
# Active learning: Bayesian experiment design (CORRECTED)
# ============================================================

class BayesianExperimentDesign:
    """
    Bayesian Experiment Design with proper initialization and evaluation.
    Uses FIXED test set and absolute uncertainty for acquisition.
    """

    def __init__(self, n_pcs, kernel="matern"):
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

    def _fit_model(self):
        self.model = SklearnGPR_ARDSE(n_pcs=self.n_pcs)
        self.model.fit(self.X_train, self.y_train)

    def _information_gain(self, X):
        # CORRECT: Use absolute uncertainty σ(x), not |σ/μ|
        mu, sigma = self.model.predict(X, return_std=True)
        return sigma, mu, sigma  # Return sigma as the information gain

    def run(self, X_all, y_all, formulas, initial_n=4, batch_size=1,
            max_samples=35, test_size=0.2, mape_threshold=2.0, random_state=42):
        """
        Run active learning loop with FIXED test set.
        X_all, y_all: Full dataset
        formulas: Array of composition strings (e.g., "Al", "Nb92Ti8")
        """
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)

        # Split into pool and fixed test set
        X_pool, X_test, y_pool, y_test, formulas_pool, formulas_test = train_test_split(
            X_all, y_all, formulas, test_size=test_size, random_state=random_state
        )

        # Initialize with ELEMENTAL SYSTEMS (pure Al, Nb, Ti, Zr)
        # Find indices of elemental compositions
        elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
        elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas_pool])

        if np.sum(elemental_mask) >= 4:
            # Use elemental systems as initial training set
            init_indices = np.where(elemental_mask)[0][:4]
            self.X_train = X_pool[init_indices].copy()
            self.y_train = y_pool[init_indices].copy()

            # Remove from pool
            pool_indices = np.setdiff1d(np.arange(len(X_pool)), init_indices)
        else:
            # Fallback: random initialization
            print("Warning: Could not find 4 elemental compositions. Using random initialization.")
            np.random.seed(random_state)
            init_indices = np.random.choice(len(X_pool), size=min(initial_n, len(X_pool)), replace=False)
            self.X_train = X_pool[init_indices].copy()
            self.y_train = y_pool[init_indices].copy()
            pool_indices = np.setdiff1d(np.arange(len(X_pool)), init_indices)

        print(f"Initial training set: {len(self.X_train)} samples")
        print(f"Pool size: {len(pool_indices)} samples")
        print(f"Test set size: {len(X_test)} samples")

        iteration = 0
        while len(self.X_train) <= max_samples and len(pool_indices) > 0:
            # Fit model on current training set
            self._fit_model()

            # Evaluate on FIXED test set (never used for selection)
            y_pred_test, y_std_test = self.model.predict(X_test, return_std=True)
            metrics = ErrorMetrics.compute_all(y_test, y_pred_test)

            self.history["n_samples"].append(len(self.X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_test)))

            print(
                f"[ACTIVE] iter={iteration:02d}, n_train={len(self.X_train):3d}, "
                f"MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}"
            )

            # Convergence check
            if metrics["MAPE"] <= mape_threshold:
                print(f"  → Converged (MAPE ≤ {mape_threshold:.2f} %)")
                break

            if len(pool_indices) == 0 or len(self.X_train) >= max_samples:
                break

            # Compute information gain on candidate pool
            # CORRECT: Use absolute uncertainty σ(x)
            ig_pool, mu_pool, sigma_pool = self._information_gain(X_pool[pool_indices])
            select_pool_idx_local = np.argsort(ig_pool)[-batch_size:]
            select_pool_idx_global = pool_indices[select_pool_idx_local]

            # Add selected samples to training set
            self.X_train = np.vstack([self.X_train, X_pool[select_pool_idx_global]])
            self.y_train = np.concatenate([self.y_train, y_pool[select_pool_idx_global]])

            # Remove selected from pool
            pool_indices = np.setdiff1d(pool_indices, select_pool_idx_global)

            iteration += 1

        # Final fit on all collected training data
        self._fit_model()
        return self.model, self.history


# ============================================================
# Random sampling baseline (CORRECTED)
# ============================================================

class RandomSamplingBaseline:
    """Random selection with same initialization and evaluation as active."""

    def __init__(self, n_pcs, kernel="matern"):
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

    def run(self, X_all, y_all, formulas, initial_n=4, batch_size=1,
            max_samples=35, test_size=0.2, random_state=123):
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)
        formulas = np.asarray(formulas)

        # Use SAME split as active learning
        X_pool, X_test, y_pool, y_test, formulas_pool, formulas_test = train_test_split(
            X_all, y_all, formulas, test_size=test_size, random_state=42  # Same split
        )

        # Use SAME elemental initialization
        elemental_symbols = ["Al", "Nb", "Ti", "Zr"]
        elemental_mask = np.array([any(f.strip() == sym for sym in elemental_symbols) for f in formulas_pool])

        if np.sum(elemental_mask) >= 4:
            init_indices = np.where(elemental_mask)[0][:4]
            X_train = X_pool[init_indices].copy()
            y_train = y_pool[init_indices].copy()
            pool_indices = np.setdiff1d(np.arange(len(X_pool)), init_indices)
        else:
            np.random.seed(random_state)
            init_indices = np.random.choice(len(X_pool), size=min(initial_n, len(X_pool)), replace=False)
            X_train = X_pool[init_indices].copy()
            y_train = y_pool[init_indices].copy()
            pool_indices = np.setdiff1d(np.arange(len(X_pool)), init_indices)

        iteration = 0
        while len(X_train) <= max_samples and len(pool_indices) > 0:
            # Fit model
            gpr = SklearnGPR_ARDSE(n_pcs=self.n_pcs)
            gpr.fit(X_train, y_train)

            # Evaluate on SAME test set as active learning
            y_pred_test, y_std_test = gpr.predict(X_test, return_std=True)
            metrics = ErrorMetrics.compute_all(y_test, y_pred_test)

            self.history["n_samples"].append(len(X_train))
            self.history["MAE"].append(metrics["MAE"])
            self.history["MAPE"].append(metrics["MAPE"])
            self.history["RMSE"].append(metrics["RMSE"])
            self.history["R2"].append(metrics["R2"])
            self.history["MeanSigma"].append(float(np.mean(y_std_test)))

            print(
                f"[RANDOM] iter={iteration:02d}, n_train={len(X_train):3d}, "
                f"MAPE={metrics['MAPE']:.2f} %, R2={metrics['R2']:.4f}"
            )

            if len(pool_indices) == 0 or len(X_train) >= max_samples:
                break

            # Randomly select from pool
            k = min(batch_size, len(pool_indices))
            select_idx_local = np.random.choice(np.arange(len(pool_indices)), size=k, replace=False)
            select_idx_global = pool_indices[select_idx_local]

            X_train = np.vstack([X_train, X_pool[select_idx_global]])
            y_train = np.concatenate([y_train, y_pool[select_idx_global]])

            pool_indices = np.setdiff1d(pool_indices, select_idx_global)

            iteration += 1

        self.model = gpr
        return self.model, self.history



def plot_convergence(active_hist, random_hist, property_name,
                     fname="Figure_6_Convergence.png"):
    """Analogue of Figure 6: convergence curves for active vs random."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Convergence Comparison ({property_name})", fontweight="bold")

    # MAPE
    ax = axes[0, 0]
    ax.plot(active_hist["n_samples"], active_hist["MAPE"], "o-",
            label="Active", color="#2E86AB")
    ax.plot(random_hist["n_samples"], random_hist["MAPE"], "s-",
            label="Random", color="#A23B72")
    ax.axhline(2.0, color="gray", linestyle="--", label="2%")
    ax.axhline(3.0, color="gray", linestyle=":", label="3%")
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("MAPE (%)", fontweight="bold")
    ax.set_title("MAPE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # R2
    ax = axes[0, 1]
    ax.plot(active_hist["n_samples"], active_hist["R2"], "o-",
            label="Active", color="#2E86AB")
    ax.plot(random_hist["n_samples"], random_hist["R2"], "s-",
            label="Random", color="#A23B72")
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_title("R²", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # MAE
    ax = axes[1, 0]
    ax.plot(active_hist["n_samples"], active_hist["MAE"], "o-",
            label="Active", color="#2E86AB")
    ax.plot(random_hist["n_samples"], random_hist["MAE"], "s-",
            label="Random", color="#A23B72")
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("MAE", fontweight="bold")
    ax.set_title("MAE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # RMSE
    ax = axes[1, 1]
    ax.plot(active_hist["n_samples"], active_hist["RMSE"], "o-",
            label="Active", color="#2E86AB")
    ax.plot(random_hist["n_samples"], random_hist["RMSE"], "s-",
            label="Random", color="#A23B72")
    ax.set_xlabel("Training samples", fontweight="bold")
    ax.set_ylabel("RMSE", fontweight="bold")
    ax.set_title("RMSE", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_predictions(y_true, y_pred, y_std, property_name,
                     fname="Figure_7_Predictions.png"):
    """Analogue of Figure 7a: predictions vs truth and residual plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Predictions vs Ground Truth ({property_name})", fontweight="bold")

    # Left: y_true vs y_pred with error bars
    ax = axes[0]
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ecolor="gray",
                capsize=3, alpha=0.7, markersize=5,
                markeredgecolor="k", markerfacecolor="#2E86AB")
    ymin, ymax = np.min(y_true), np.max(y_true)
    ax.plot([ymin, ymax], [ymin, ymax], "r--", label="1:1 line")
    ax.set_xlabel("Ground truth", fontweight="bold")
    ax.set_ylabel("Prediction", fontweight="bold")
    ax.set_title("Predictions with uncertainty", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_box_aspect(1)

    # Right: residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, s=30, alpha=0.8, edgecolor="k",
               facecolor="#A23B72")
    ax.axhline(0.0, color="r", linestyle="--")
    # ±2σ band based on mean sigma
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
    plt.close(fig)
    print(f"Saved {fname}")


def plot_uncertainty_hist(y_true, y_pred, y_std, property_name,
                          fname="Figure_7b_Uncertainty.png"):
    """Analogue of Figure 7b: histogram of σ and coverage statistics."""
    errors = np.abs(y_true - y_pred)
    within_1 = np.mean(errors <= y_std) * 100.0
    within_2 = np.mean(errors <= 2.0 * y_std) * 100.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_std, bins=30, color="#2E86AB", alpha=0.8,
            edgecolor="k")
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
    ax.legend()
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Saved {fname}")


# ============================================================
# Main script
# ============================================================

df = pd.read_csv('pca_pspall7.csv')#df_pca.copy()

# Check if we need to find the REAL data file
if len(df) < 1000:  # If it's just a small example
    print("Looking for full dataset with 4495 samples...")
    possible_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'psp' in f]
    if len(possible_files) > 1:
        csv_path = max(possible_files, key=lambda x: os.path.getsize(x))
        print(f"Found full dataset: {csv_path} ({os.path.getsize(csv_path)} bytes)")
        df = pd.read_csv(csv_path)

df = df.reset_index(drop=True)

# Verify columns exist
required_cols = ["bulk_modulus", "formation_energy"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    if df[col].isna().any():
        raise ValueError(f"Column {col} has NaN values. Please provide complete dataset.")

# PC columns
pc_cols = [c for c in df.columns if c.startswith("PC")]
if len(pc_cols) < 3:
    raise ValueError("Need at least PC1, PC2, PC3 in the CSV.")

# Use first 3 PCs as features
n_pcs = 3
X = df[pc_cols[:n_pcs]].values
y_bm = df["bulk_modulus"].values
y_fe = df["formation_energy"].values

# Get formula column for elemental initialization
formula_col = None
for col in ["formula", "composition", "comp", "Formula"]:
    if col in df.columns:
        formula_col = col
        break

if formula_col is None:
    print("Warning: No formula column found. Will use random initialization.")
    formulas = np.array([f"Sample{i}" for i in range(len(df))])
else:
    formulas = df[formula_col].values

print(f"✓ Loaded full dataset: {len(df)} samples")
print(f"Feature matrix shape: {X.shape}")
print(f"Bulk modulus range: {y_bm.min():.2f} - {y_bm.max():.2f} GPa")
print(f"Formation energy range: {y_fe.min():.2f} - {y_fe.max():.2f} eV/atom")

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# --- Figure 3: PCA variance explained ---
plot_figure3_pca_variance(df, pc_cols)

# --- Figure 5: PC space visualization ---
if len(df) >= 100:
    plot_figure5_pc_space(df, pc_cols, target_col="bulk_modulus")

print("\n" + "="*80)
print("BULK MODULUS - ACTIVE vs RANDOM (Fixed Test Set, Elemental Init)")
print("="*80)

# BULK MODULUS
active_bm = BayesianExperimentDesign(n_pcs=n_pcs, kernel="matern")
model_bm_active, hist_bm_active = active_bm.run(
    X_scaled, y_bm, formulas,
    initial_n=4,
    batch_size=1,
    max_samples=35,
    test_size=0.9,
    mape_threshold=2.0,
    random_state=42
)

random_bm = RandomSamplingBaseline(n_pcs=n_pcs, kernel="matern")
model_bm_rand, hist_bm_rand = random_bm.run(
    X_scaled, y_bm, formulas,
    initial_n=4,
    batch_size=1,
    max_samples=35,
    test_size=0.9,  # Same test set
    random_state=123
)

# Final predictions on full dataset
y_bm_pred, y_bm_std = active_bm.model.predict(X_scaled, return_std=True)
metrics_bm = ErrorMetrics.compute_all(y_bm, y_bm_pred)
print("\nFinal Bulk Modulus metrics (Active selection):")
for k, v in metrics_bm.items():
    if k in ("MAPE", "NMAE"):
        print(f"  {k:10s}: {v:8.3f} %")
    else:
        print(f"  {k:10s}: {v:8.4f}")

# Plots for bulk modulus
plot_convergence(
    hist_bm_active, hist_bm_rand,
    property_name="Bulk Modulus",
    fname="Figure_6_Convergence_BulkModulus.png"
)
plot_predictions(
    y_bm, y_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Figure_7_Predictions_BulkModulus.png"
)
plot_uncertainty_hist(
    y_bm, y_bm_pred, y_bm_std,
    property_name="Bulk Modulus",
    fname="Figure_7b_Uncertainty_BulkModulus.png"
)

print("\n" + "="*80)
print("FORMATION ENERGY - ACTIVE vs RANDOM (Fixed Test Set, Elemental Init)")
print("="*80)

# FORMATION ENERGY
active_fe = BayesianExperimentDesign(n_pcs=n_pcs, kernel="matern")
model_fe_active, hist_fe_active = active_fe.run(
    X_scaled, y_fe, formulas,
    initial_n=4,
    batch_size=1,
    max_samples=35,
    test_size=1,
    mape_threshold=2.0,
    random_state=99
)

random_fe = RandomSamplingBaseline(n_pcs=n_pcs, kernel="matern")
model_fe_rand, hist_fe_rand = random_fe.run(
    X_scaled, y_fe, formulas,
    initial_n=4,
    batch_size=1,
    max_samples=35,
    test_size=1,  # Same test set
    random_state=777
)

# Final predictions for formation energy
y_fe_pred, y_fe_std = active_fe.model.predict(X_scaled, return_std=True)
metrics_fe = ErrorMetrics.compute_all(y_fe, y_fe_pred)
print("\nFinal Formation Energy metrics (Active selection):")
for k, v in metrics_fe.items():
    if k in ("MAPE", "NMAE"):
        print(f"  {k:10s}: {v:8.3f} %")
    else:
        print(f"  {k:10s}: {v:8.4f}")

# Plots for formation energy
plot_convergence(
    hist_fe_active, hist_fe_rand,
    property_name="Formation Energy",
    fname="Figure_6_Convergence_FormationEnergy.png"
)
plot_predictions(
    y_fe, y_fe_pred, y_fe_std,
    property_name="Formation Energy",
    fname="Figure_7_Predictions_FormationEnergy.png"
)
plot_uncertainty_hist(
    y_fe, y_fe_pred, y_fe_std,
    property_name="Formation Energy",
    fname="Figure_7b_Uncertainty_FormationEnergy.png"
)

# Combined metrics comparison table
rand_bm_pred = random_bm.model.predict(X_scaled, return_std=False)
rand_fe_pred = random_fe.model.predict(X_scaled, return_std=False)
metrics_bm_rand = ErrorMetrics.compute_all(y_bm, rand_bm_pred)
metrics_fe_rand = ErrorMetrics.compute_all(y_fe, rand_fe_pred)

rows = []
rows.append({
    "Property": "Bulk Modulus",
    "Strategy": "Active",
    **metrics_bm
})
rows.append({
    "Property": "Bulk Modulus",
    "Strategy": "Random",
    **metrics_bm_rand
})
rows.append({
    "Property": "Formation Energy",
    "Strategy": "Active",
    **metrics_fe
})
rows.append({
    "Property": "Formation Energy",
    "Strategy": "Random",
    **metrics_fe_rand
})

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("metrics_comparison.csv", index=False)
print("\nSaved metrics_comparison.csv:")
print(metrics_df.to_string(index=False))


