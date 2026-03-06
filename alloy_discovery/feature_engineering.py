"""
Auto-generated module from Jupyter Notebook
"""

import sys
import numpy as np
import math
import h5py
import os
import re
import pyvista as pv
import matplotlib.pyplot as plt

DIR='/content/drive'
HOME = os.path.join(DIR, 'MyDrive')
HOMED = os.path.join(HOME, "Analyze4RHEA")
os.chdir(HOMED)

#PyVista 3D numpy array plot code
def plotvista(data_3d,filex="None"):
  # Seems that only static plotting is supported by colab at the moment
    pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.notebook = True
    pv.start_xvfb()
    #setting theme
    #pv.set_plot_theme("document")

    #setting title
    pv.global_theme.title = filex

    # Create the spatial reference
    grid = pv.ImageData()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(np.shape(data_3d)) + 1

    # Edit the spatial reference
    #grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data[filex] = data_3d.flatten(order="F")  # Flatten the array!

    #grid = grid.threshold([0.000000000000000000000001, 500])

    # Now plot the grid!
    grid.plot(show_edges=False, zoom=0.7)

def slicedplot(data3d,filex="Slice of 2 point stats"):
    pv.global_theme.jupyter_backend = 'static'
    pv.global_theme.notebook = True
    pv.start_xvfb()
    #pv.set_plot_theme("document")
    #setting title
    pv.global_theme.title = filex
    # Create the spatial reference
    grid = pv.ImageData()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(data3d.shape) + 1
    # Edit the spatial reference
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
    # Add the data values to the cell data

    print(np.shape(data3d.flatten(order="F")))

    grid.cell_data[filex] = data3d.flatten(order="F")  # Flatten the array!
    slices=grid.slice_orthogonal()
    slices.plot()

def tptstats(rho_s):
    denominator=rho_s.size
    ms = np.sqrt(rho_s)
    ps=np.real(np.fft.ifftn(np.multiply(np.conj(np.fft.fftn(ms)),np.fft.fftn(ms)))) / denominator
    shifted_ps=np.fft.ifftshift(ps)
    return shifted_ps

def readPseudopot(S, ityp, psdfname,print_debug  : bool = False):
    pseudopotential_filename = os.path.join(S["inputfile_path"], psdfname)

    # pseudopotential_filename = filename

    with open(pseudopotential_filename,"r") as psp:
        # Read all lines from the file
        lines = psp.readlines()

        # Process each line: split by the separator pattern (,\s+)+(\+\S)
        # This pattern matches: comma followed by whitespace, then a plus sign with something
        separator_pattern = r'(,\s+)+(\+\S)'
        l1_list = []
        for line in lines:
            # Split by the separator pattern
            parts = re.split(separator_pattern, line)
            # Keep only the first part (before the separator) and filter out empty strings
            if parts:
                l1_list.append([parts[0].strip()])

        # Now split each string by comma or whitespace
        pattern = r',|\s+'
        l1_split = [re.split(pattern, string[0]) for string in l1_list]
        # Filter out empty strings from each split result
        l1_split = [[item for item in split_list if item] for split_list in l1_split]

        Zatom = float(l1_split[1][0])
        Z = float(l1_split[1][1])

        pspxc = float(l1_split[2][1])

        lmax = int(l1_split[2][2])
        lloc = float(l1_split[2][3])
        mmax = int(l1_split[2][4])

        fchrg = float(l1_split[3][1])

        nproj = [int(l1_split[4][i]) for i in range(int(lmax+1))]

        extension_switch = float(l1_split[5][0])
        pspsoc = 0   # indicating if the psp file including spin-orbit coupling
        nprojso = 0

        if extension_switch == 2 or extension_switch == 3:
            if print_debug:
                print("This psp8 includes spin-orbit coupling.\n")
            pspsoc = 1
            nprojso = [float(l1_split[6][i]) for i in range(int(lmax))]


        Pot   = [dict({'gamma_Jl' : np.zeros((int(nproj[i]),1)), 'proj' : (int(nproj[i]),mmax) }) for i in range(lmax+1)]
        Potso = [dict({'gamma_Jl' : np.zeros((int(nproj[i]),1)), 'proj' : (int(nproj[i]),mmax) }) for i in range(lmax+1)]

        l_read = float(l1_split[6][0])

        l_count = 0
        lc_count = 6
        for l in range(int(lmax+1)):
           if l != lloc:
               Pot[l]['gamma_Jl'] = [float(l1_split[6 + l*mmax+l][i]) for i in range(1,nproj[l]+1)]
               sz1 = (mmax,2+nproj[l])
               A1 = [float(l1_split[j+l*mmax][k]) if l==0 else float(l1_split[j+l*mmax + l][k]) for j in range(7 ,mmax+7) for k in range(2+nproj[l])]
               y1 = np.reshape(np.array(A1), sz1)
               r = y1[:,1]
               Pot[l]['proj'] = y1[:,2:]
               Pot[l]['proj'][1:,:] = Pot[l]['proj'][1:,:]/np.reshape(r[1:],(-1,1))
               Pot[l]['proj'][0,:] = Pot[l]['proj'][1,:]
           else:
                A2 = [float(l1_split[j+l*mmax][k]) if l==0 else float(l1_split[j+l*mmax + l][k]) for j in range(7 ,mmax+7) for k in range(3)]
                y2 = np.reshape(np.array(A2), (mmax, 3))
                r = y2[:,1]
                Vloc = y2[:,2]

           l_read = float(l1_split[6 + (l+1)*mmax+l+1][0])

           l_count = l_count+1
           if l ==0:
               lc_count = mmax+lc_count
           else:
               lc_count = mmax+lc_count+1


        if lloc > lmax or l_read ==4:
           A3 = [float(l1_split[j+(l_count)*mmax+l_count][k]) for j in range(7 ,mmax+7) for k in range(3)]
           y3 = np.reshape(np.array(A3), (mmax, 3))
           r = y3[:,1]
           Vloc = y3[:,2]
           l_count = l_count+1
           lc_count = mmax+lc_count+1

        '''read spin-orbit projectors'''
        if pspsoc == 1:
           for l in range(1,lmax+1):
               Potso[l]['gamma_Jl'] = [float(l1_split[6 + (l_count)*mmax][i]) for i in range(1,nproj[l]+1)]
               sz = (mmax,2+nprojso[l])
               A4 = [float(l1_split[j+(l_count)*mmax+l_count][k]) for j in range(7 ,mmax+7) for k in range(2+nprojso[l])]
               y4 = np.reshape(np.array(A4), sz)
               r = y4[:,1]
               Potso[l]['proj'] = y4[:,2:]
               Potso[l]['proj'][1:,:] = Potso[l]['proj'][1:,:]/np.reshape(r[1:],(-1,1))
               Potso[l]['proj'][0,:] = Potso[l]['proj'][1,:]
               lc_count = mmax + lc_count

        '''read core density'''
        if fchrg > 0:
            Atilde = [float(l1_split[lc_count+1+j][k]) for j in range(mmax) for k in range(7)]
            y4 = np.reshape(np.array(Atilde), (mmax, 7))
            uu = y4[:,2]/(4*math.pi)
            rho_tilde = uu
            rTilde = y4[:,1]
            lc_count = mmax + lc_count

        else:
            rTilde = r
            rho_tilde= np.zeros((np.size(r),1))

        uu = np.zeros((mmax))
        A5 = [float(l1_split[lc_count+1+j][k]) for j in range(mmax) for k in range(5)]
        y5 = np.reshape(np.array(A5), (mmax,5))
        uu[:] = y5[:,2]/(4*math.pi)
        rho_isolated_guess = uu

        rc = 0

        rc_max_list = np.zeros((lmax+1))
        for l in range(lmax+1):
            r_core_read = float(l1_split[0][l+3])
            rc_max = r_core_read
            if l != lloc:
                ''' % check if r_core is large enough s.t. |proj| < 1E-8'''
                r_indx_all = np.where(r < r_core_read)
                r_indx = r_indx_all[0][-1]
                for i in range(np.shape(Pot[l]['proj'])[1]):
                    try:
                        rc_temp = r[r_indx +np.where(np.absolute(Pot[l]['proj'][r_indx+1:,i])<(1e-8))[0][0] - 1]
                    except:
                        rc_temp = r[-1]
                    if rc_temp>rc_max:
                        rc_max = rc_temp

                    rc_max_list[l] = rc_max
                if print_debug:
                    print("atom type {first}, l = {second}, r_core_read {third}, change to rmax where |UdV| < (1e-8), {fourth} \n".format(first = 1, second = l, third = r_core_read, fourth = rc_max))
        if rc_max > rc:
            rc = rc_max

        r_grid_vloc = r
        r_grid_rho = r

        XC = pspxc
        r_grid_rho_Tilde = rTilde

        # return [Z,
        # Zatom,
        # XC,
        # Vloc,
        # r_grid_vloc,
        # rc,
        # Pot,
        # lmax,
        # lloc,
        # nproj,
        # r_grid_rho,
        # rho_isolated_guess,
        # rho_tilde,
        # r_grid_rho_Tilde,
        # pspsoc,
        # Potso,
        # rc_max_list]

        # ----------------------------------------------------------------------
        # Assign to S
        # ----------------------------------------------------------------------
        S["Atm"][ityp]["Z"] = Z
        S["Atm"][ityp]["Zatom"] = int(Zatom)
        S["Atm"][ityp]["Vloc"] = Vloc
        S["Atm"][ityp]["r_grid_vloc"] = r_grid_vloc
        S["Atm"][ityp]["Pot"] = Pot
        S["Atm"][ityp]["Potso"] = Potso
        S["Atm"][ityp]["lmax"] = lmax
        S["Atm"][ityp]["lloc"] = lloc
        S["Atm"][ityp]["nproj"] = nproj
        S["Atm"][ityp]["nprojso"] = nprojso
        S["Atm"][ityp]["rho_Tilde"] = rho_tilde
        S["Atm"][ityp]["rho_isolated_guess"] = rho_isolated_guess
        S["Atm"][ityp]["r_grid_rho_Tilde"] = r_grid_rho_Tilde
        S["Atm"][ityp]["r_grid_rho"] = r_grid_rho
        S["Atm"][ityp]["pspsoc"] = pspsoc
        S["Atm"][ityp]["rc"] = rc

    return S



import numpy as np

def setup_nonorthogonal_cell(S):
    """
    Checks if the cell is non-orthogonal and calculates the necessary
    transformation matrices (metric, gradient, Laplacian) and converts
    atomic coordinates if S['cell_typ'] is set to 2.

    Args:
        S (dict): The main structure/dictionary containing simulation parameters.
                  Must contain 'lat_vec', 'temp_tol', 'n_typ', 'IsFrac', 'Atm', and 'Atoms'.

    Returns:
        dict: The updated structure S.
    """

    # Ensure S['cell_typ'] is initialized (assuming 1 if not set)
    if 'cell_typ' not in S:
        S['cell_typ'] = 1

    # Get lattice vectors (lat_vec is assumed to be 3x3)
    # The MATLAB dot function uses dot(A(i,:), A(j,:)) for row-wise dot product.
    lat_vec = np.array(S['lat_vec'])
    temp_tol = S['temp_tol']

    ## 1. Check Cell Type (Orthogonal vs. Non-Orthogonal)

    # Check if any dot product of basis vectors (rows) is non-zero (non-orthogonal)
    # The dot product of row i and row j is lat_vec[i] @ lat_vec[j]

    # MATLAB: abs(dot(S.lat_vec(1,:),S.lat_vec(2,:))) > S.temp_tol || ...
    if (np.abs(np.dot(lat_vec[0, :], lat_vec[1, :])) > temp_tol or
        np.abs(np.dot(lat_vec[1, :], lat_vec[2, :])) > temp_tol or
        np.abs(np.dot(lat_vec[2, :], lat_vec[0, :])) > temp_tol):

        S['cell_typ'] = 2

    ## 2. Calculate Unit Lattice Vectors (lat_uvec)

    # Initialize lat_uvec (3x3 array)
    S['lat_uvec'] = np.zeros_like(lat_vec)

    # Calculate unit vectors
    for i in range(3):
        # norm(S.lat_vec(i+1,:))
        norm_i = np.linalg.norm(lat_vec[i, :])
        # S.lat_uvec(i+1,:) = S.lat_vec(i+1,:)/norm(...)
        S['lat_uvec'][i, :] = lat_vec[i, :] / norm_i

    ## 3. Set up Transformation Matrices (if non-orthogonal)

    if S['cell_typ'] == 2:
        lat_uvec = S['lat_uvec']

        # NOTE on MATLAB vs. NumPy:
        # MATLAB's `lat_uvec'` means transpose (matrix conjugation)
        # NumPy's `lat_uvec.T` means transpose.

        # Jacobian (Volume of the unit cell formed by unit vectors)
        # S.Jacb = det(S.lat_uvec');
        S['Jacb'] = np.linalg.det(lat_uvec.T)

        if S['Jacb'] <= 0.0:
            raise AssertionError("Volume is zero or negative!")

        # Metric Tensor: metric_T = lat_uvec * lat_uvec'
        # This is G_{ij} = a_i \cdot a_j where a_i are the unit vectors
        S['metric_T'] = lat_uvec @ lat_uvec.T

        # Multiply off-diagonal terms by 2 (Likely for internal use in a specific FD scheme)
        # MATLAB indices (1,2), (2,3), (1,3) => Python indices (0,1), (1,2), (0,2)
        S['metric_T'][0, 1] *= 2
        S['metric_T'][1, 2] *= 2
        S['metric_T'][0, 2] *= 2

        # Gradient Transformation Matrix: grad_T = inv(lat_uvec')
        S['grad_T'] = np.linalg.inv(lat_uvec.T)

        # Laplacian Transformation Matrix: lapc_T = grad_T * grad_T'
        S['lapc_T'] = S['grad_T'] @ S['grad_T'].T

        # Multiply off-diagonal terms by 2
        S['lapc_T'][0, 1] *= 2
        S['lapc_T'][1, 2] *= 2
        S['lapc_T'][0, 2] *= 2

        ## 4. Convert Atomic Coordinates (if non-orthogonal)

        count_prev = 0
        count = 0

        for ityp in range(S['n_typ']): # ityp runs from 0 to n_typ - 1 in Python
            # Access the atom type data dictionary
            Atm_typ = S['Atm'][ityp]

            # The MATLAB code seems to use S.Atm as a list/array of structs indexed from 1
            # We assume S['Atm'] is a list of dictionaries, indexed from 0 (ityp).

            if Atm_typ.get('IsFrac', 0) == 0: # Check if coordinates are Cartesian (non-fractional)

                # S.Atm(ityp).coords = transpose(S.grad_T * transpose(S.Atm(ityp).coords));
                # This performs a change of basis (transformation) on the atomic coordinates

                # Get coordinates (assumed N_atm_typ x 3)
                coords = np.array(Atm_typ['coords'])

                # Python: (grad_T @ coords.T).T (equivalent to MATLAB's transpose(grad_T * transpose(coords)))
                transformed_coords = (S['grad_T'] @ coords.T).T

                Atm_typ['coords'] = transformed_coords # Update in Atm structure

                # Update global S.Atoms list
                n_atm_typ = Atm_typ['n_atm_typ']
                count = count + n_atm_typ

                # S.Atoms(count_prev+1:count,:) = S.Atm(ityp).coords;
                # Convert 1-based indexing to 0-based
                S['Atoms'][count_prev:count, :] = transformed_coords

                count_prev = count

            else:
                # Fractional coordinates are not transformed here, only counters updated
                n_atm_typ = Atm_typ['n_atm_typ']
                count = count + n_atm_typ
                count_prev = count

    return S

from ase.io import read
from ase.units import Bohr
from ase.data import atomic_numbers # Used to map symbol to atomic number
import numpy as np
from collections import Counter
from math import ceil
from ase.data import atomic_masses

def read_and_store_info(S, fname):
    # --- 1. Read the POSCAR file ---
    # ASE automatically reads the cell vectors and atom positions in Ångstrom (Å),
    # which is the standard unit for POSCAR files.
    atoms = read(fname)

    # --- 2. Convert Cell to Bohr ---
    # The cell is a 3x3 matrix. Multiply the cell matrix by the conversion factor.
    cell_angstrom = atoms.get_cell()
    cell_bohr = cell_angstrom / Bohr

    S['lat_vec'] = cell_bohr # Store 3x3 lattice vector matrix
    S['L1'], S['L2'], S['L3'] = np.linalg.norm(cell_bohr, axis=1) # Get vector magnitudes

    # --- 3. Convert Atom Coordinates to Bohr ---
    # The positions are an Nx3 array. Multiply the positions array by the factor.
    positions_angstrom = atoms.get_positions()
    positions_bohr = positions_angstrom / Bohr

    n_atm = len(positions_bohr)
    S['n_atm'] = n_atm

    # 2. Extract Atomic Data
    atom_symbols = atoms.get_chemical_symbols()
    # Get unique element types and their counts as ordered lists
    unique_elements, counts = np.unique(atom_symbols, return_counts=True)

    type_counts = Counter(atom_symbols)
    # Get the ordered list of unique types and their counts based on the order in the file
    ordered_types = []
    ordered_counts = []
    seen = set()
    for symbol in atom_symbols:
        if symbol not in seen:
            ordered_types.append(symbol)
            ordered_counts.append(type_counts[symbol])
            seen.add(symbol)

    S['n_typ'] = len(ordered_types) # Number of unique atom types

    # Check the coordinate type used in the POSCAR
    # ASE stores the coordinate type used in the file
    input_coords_type = atoms.info.get('coordinate_type', 'Cartesian')

    # IsFrac = 1 for Fractional (Direct), 0 for Cartesian (Conventional)
    if input_coords_type.lower().startswith(('d', 'direct')):
        IsFrac_flag = 1
        print("   -> Coordinates detected as Fractional (Direct).")
    else:
        IsFrac_flag = 0
        print("   -> Coordinates detected as Cartesian (Conventional).")

    # 4. Populate S['Atoms'] and S['Atm']
    S['Atoms'] = positions_bohr # Global list of all atom positions (N_atm x 3)
    S['Atm'] = []
    current_index = 0

    # Initialize global constraint array (default to fully relax: 1 1 1)
    S['mvAtmConstraint'] = np.ones((n_atm, 3), dtype=int)

    # Initialize flag vectors (used later in the main script)
    S['IsFrac'] = np.zeros(S['n_typ'], dtype=int)
    S['IsSpin'] = np.zeros(S['n_typ'], dtype=int)

    for ityp, (symbol, count) in enumerate(zip(ordered_types, ordered_counts)):
        # Get the Atomic Number (Z_atom) using ASE's data module
        Z_atom = atomic_numbers[symbol]

        # Slice the global positions array to get coordinates for the current type
        coords_for_type = positions_bohr[current_index : current_index + count]

        atm_type_entry = {
            'element': symbol,
            'n_atm_typ': count,
            'coords': coords_for_type,       # N_type x 3 array in Bohr
            'IsFrac': IsFrac_flag                      # Assuming Cartesian (0)
            # Z, Vloc, etc., will be added by read_pseudopot()
        }

        S['Atm'].append(atm_type_entry)

        Atm_typ = S['Atm'][ityp]

        elemType = Atm_typ['element']
        # Z_atom = Atm_typ['Zatom']
        n_atm_typ = Atm_typ['n_atm_typ']

        # 1. Construct the filename based on the rule: Atomic_number.psp8
        # Format Z_atom with leading zeros (e.g., 8 -> '08')
        psdfname = f"./psp8_SPMS_GGA/{Z_atom:02d}.psp8"
        Atm_typ['psdfname'] = psdfname

        # --- Set Other Default Parameters ---

        Atm_typ['typ'] = elemType

        # Default atomic mass from ASE (atomic_masses are indexed by Z_atom)
        Atm_typ['Mass'] = atomic_masses[Z_atom]
        #print(f" Default atomic mass for {elemType} (Z={Z_atom}) is {Atm_typ['Mass']:.6f} a.u.")

        # Default PSP properties
        Atm_typ['lloc'] = 4
        Atm_typ['psptyp'] = 1
        Atm_typ['IsFrac'] = Atm_typ.get('IsFrac', 0) # Read from POSCAR

        # Update IsFrac flag array
        S['IsFrac'][ityp] = Atm_typ['IsFrac']

        # Explicitly set the path flag if a psdfname was generated
        S['is_default_psd'] = 0

        # Atm_typ = S['Atm'][ityp]
        # Z_atom = Atm_typ['Zatom']

        # 2. Call the core pseudopotential reading function (the one you provided)
        # Note: We must pass the correct arguments: S, ityp (index), and psdfname (string).
        S = readPseudopot(S, ityp, psdfname)

        current_index += count

    # Calculate Total Mass
    S['TotalMass'] = 0.0
    for Atm_typ in S['Atm']:
        S['TotalMass'] += Atm_typ['Mass'] * Atm_typ['n_atm_typ']

    # Check if non-orthogonal cell
    S = setup_nonorthogonal_cell(S)

    # Convert mesh spacing to atomic units
    h = S['mesh_spacing']/Bohr

    # For periodic systems only
    S['Nx'] = max(ceil(S['L1'] / h), S['FDn'])
    S['Ny'] = max(ceil(S['L2'] / h), S['FDn'])
    S['Nz'] = max(ceil(S['L3'] / h), S['FDn'])
    S['dx'] = S['L1'] / S['Nx']
    S['dy'] = S['L2'] / S['Ny']
    S['dz'] = S['L3'] / S['Nz']
    S['N'] = S['Nx']*S['Ny']*S['Nz']

    S = calculate_weights(S)

    return S, atoms.symbols

def IntgWts(Nx, Ny, Nz, BCx, BCy, BCz, xin, S):
    """
    Calculates the integration weights (W) vector for a 3D grid,
    applying boundary condition corrections for Cartesian/Non-Orthogonal cells.
    """

    cell_typ = S.get('cell_typ', 1)

    if cell_typ == 1 or cell_typ == 2:

        # Calculate 1D Weights in X direction
        W_x = np.ones(Nx) * S['dx']
        W_x[0] = W_x[0] * (1 - BCx * 0.5)
        W_x[Nx - 1] = W_x[Nx - 1] * (1 - BCx * 0.5)

        # Calculate 1D Weights in Y direction
        W_y = np.ones(Ny) * S['dy']
        W_y[0] = W_y[0] * (1 - BCy * 0.5)
        W_y[Ny - 1] = W_y[Ny - 1] * (1 - BCy * 0.5)

        # Calculate 1D Weights in Z direction
        W_z = np.ones(Nz) * S['dz']
        W_z[0] = W_z[0] * (1 - BCz * 0.5)
        W_z[Nz - 1] = W_z[Nz - 1] * (1 - BCz * 0.5)

        # Construct 3D Weights using Kronecker Product and Jacobian
        # Jacb is the determinant of the transformation matrix, acting as a volume scaling factor.
        W_y_x = np.kron(W_y, W_x)
        W = np.kron(W_z, W_y_x) * S.get('Jacb', 1.0)

        # Return as a column vector
        return W.reshape(-1, 1)

    else:
        raise NotImplementedError(f"Cell type {cell_typ} is not implemented.")


from math import factorial
def calculate_weights(S):
    """
    Calculates the Finite Difference weights (w1, w2) and the integration
    weights (W) for the simulation grid.

    Args:
        S (dict): The main structure containing simulation parameters (FDn, dx, Jacb, etc.).

    Returns:
        dict: The updated S structure with 'w1', 'w2', and 'W' arrays.
    """

    FDn = S['FDn']

    # --- Finite difference weights of the second derivative (w2) ---
    w2 = np.zeros(FDn + 1)

    for k in range(1, FDn + 1):
        # Calculate the off-center weights w2[k+1] (MATLAB k+1 = Python k)
        w2[k] = (2 * (-1)**(k + 1) * (factorial(FDn)**2) /
                 (k * k * factorial(FDn - k) * factorial(FDn + k)))

        # Accumulate the central weight w2[1] (MATLAB w2(1) = Python w2[0])
        w2[0] = w2[0] - 2 * (1 / (k * k))

    S['w2'] = w2

    # --- Finite difference weights of the first derivative (w1) ---
    w1 = np.zeros(FDn + 1)

    # w1[0] is always 0 for the central point
    for k in range(1, FDn + 1):
        w1[k] = ((-1)**(k + 1) * (factorial(FDn)**2) /
                (k * factorial(FDn - k) * factorial(FDn + k)))

    S['w1'] = w1

    # --- Weights for spatial integration over domain (W) ---
    Jacb = S.get('Jacb', 1.0)

    if S['cell_typ'] == 1 or S['cell_typ'] == 2:
        # For Cartesian (1) or Non-Orthogonal (2), calculate the uniform volume element.
        uniform_volume = S['dx'] * S['dy'] * S['dz'] * Jacb
        # This assumes periodic BCs and uses the uniform volume element for the core grid points.
        S['W'] = np.full(S['N'], uniform_volume).reshape(-1, 1)
    else:

        # NOTE: The definition for IntgWts must be available in the environment.
        S['W'] = IntgWts(S['Nx'], S['Ny'], S['Nz'], S['BCx'], S['BCy'], S['BCz'], S['xin'], S)

    return S

import numpy as np
from scipy.interpolate import interp1d

def pseudochargeDensity_atom(V, II, JJ, KK, xin, S):
    """
    Compute pseudocharge density: b = laplacian(V)
    V: potential array
    II, JJ, KK: 1D index arrays of interior nodes
    xin: reference coordinate (unused)
    S: dict with FD parameters
    """
    b = np.zeros_like(V)

    if S["cell_typ"] == 1:
        dx2 = S["dx"]**2
        dy2 = S["dy"]**2
        dz2 = S["dz"]**2
        coeff = S["w2"][0] * (1/dx2 + 1/dy2 + 1/dz2)

        tgt = np.ix_(II, JJ, KK)
        b[tgt] = coeff * V[tgt]

        for p in range(1, S["FDn"] + 1):
            b[tgt] += (
                S["w2"][p]/dx2 * (V[np.ix_(II+p, JJ, KK)] + V[np.ix_(II-p, JJ, KK)]) +
                S["w2"][p]/dy2 * (V[np.ix_(II, JJ+p, KK)] + V[np.ix_(II, JJ-p, KK)]) +
                S["w2"][p]/dz2 * (V[np.ix_(II, JJ, KK+p)] + V[np.ix_(II, JJ, KK-p)])
            )

    elif S["cell_typ"] == 2:
        dx2 = S["dx"]**2
        dy2 = S["dy"]**2
        dz2 = S["dz"]**2
        dxdy = S["dx"] * S["dy"]
        dydz = S["dy"] * S["dz"]
        dzdx = S["dz"] * S["dx"]
        lapc = S["lapc_T"]
        tgt = np.ix_(II, JJ, KK)

        coeff = S["w2"][0] * (lapc[0,0]/dx2 + lapc[1,1]/dy2 + lapc[2,2]/dz2)
        b[tgt] = coeff * V[tgt]

        for p in range(1, S["FDn"] + 1):
            # main axes
            b[tgt] += (
                S["w2"][p]*lapc[0,0]/dx2 * (V[np.ix_(II+p, JJ, KK)] + V[np.ix_(II-p, JJ, KK)]) +
                S["w2"][p]*lapc[1,1]/dy2 * (V[np.ix_(II, JJ+p, KK)] + V[np.ix_(II, JJ-p, KK)]) +
                S["w2"][p]*lapc[2,2]/dz2 * (V[np.ix_(II, JJ, KK+p)] + V[np.ix_(II, JJ, KK-p)])
            )
            # cross terms
            for q in range(1, S["FDn"] + 1):
                b[tgt] += (
                    S["w1"][p]*S["w1"][q]*lapc[0,1]/dxdy *
                    (V[np.ix_(II+q, JJ+p, KK)] - V[np.ix_(II-q, JJ+p, KK)] -
                     V[np.ix_(II+q, JJ-p, KK)] + V[np.ix_(II-q, JJ-p, KK)]) +

                    S["w1"][p]*S["w1"][q]*lapc[1,2]/dydz *
                    (V[np.ix_(II, JJ+q, KK+p)] - V[np.ix_(II, JJ-q, KK+p)] -
                     V[np.ix_(II, JJ+q, KK-p)] + V[np.ix_(II, JJ-q, KK-p)]) +

                    S["w1"][p]*S["w1"][q]*lapc[0,2]/dzdx *
                    (V[np.ix_(II+q, JJ, KK+p)] - V[np.ix_(II-q, JJ, KK+p)] -
                     V[np.ix_(II+q, JJ, KK-p)] + V[np.ix_(II-q, JJ, KK-p)])
                )


    return b



import numpy as np

def calculateDistance(X, Y, Z, X_ref, Y_ref, Z_ref, S):
    """
    SPARC-consistent distance computation with periodic minimum-image
    and correct metric usage.
    """

    # Cartesian cell
    if S["cell_typ"] == 1:
        XX = X - X_ref
        YY = Y - Y_ref
        ZZ = Z - Z_ref
        return np.sqrt(XX**2 + YY**2 + ZZ**2)

    # Orthorhombic with metric tensor
    elif S["cell_typ"] == 2:
        XX = X - X_ref
        YY = Y - Y_ref
        ZZ = Z - Z_ref

        g = S["metric_T"]

        dd2 = (
            g[0,0]*XX**2 +
            g[1,1]*YY**2 +
            g[2,2]*ZZ**2 +
            2*g[0,1]*XX*YY +
            2*g[0,2]*XX*ZZ +
            2*g[1,2]*YY*ZZ
        )
        return np.sqrt(dd2)

    # Cylindrical/spherical
    elif S["cell_typ"] in (3, 4, 5):
        rho = np.sqrt((X-X_ref)**2 + (Y-Y_ref)**2)
        z   = Z - Z_ref
        return np.sqrt(rho**2 + z**2)

    else:
        raise ValueError("Unknown cell_typ")


def Calculate_rb(S):
    """
    Compute pseudocharge radius for each atom type.
    """

    if S["cell_typ"] in (1, 2):
        pos_atm_x = 0.0
        pos_atm_y = 0.0
        pos_atm_z = 0.0
        rb_up_x = (10 + 10*S["dx"]) if S["dx"] < 1.5 else (20*S["dx"] - 9.5)
        rb_up_y = (10 + 10*S["dy"]) if S["dy"] < 1.5 else (20*S["dy"] - 9.5)
        rb_up_z = (10 + 10*S["dz"]) if S["dz"] < 1.5 else (20*S["dz"] - 9.5)
        f_rby = lambda y: y
    elif S["cell_typ"] in (3, 4, 5):
        pos_atm_x = S["xmax_at"]
        pos_atm_y = 0.0
        pos_atm_z = 0.0
        rb_up_x = S["xvac"]
        f_rby = lambda y: np.arccos(1 - y**2/(2*pos_atm_x**2))
        rb_up_y = f_rby(12)
        rb_up_z = 12

    # start/end indices
    ii_s_temp = -int(np.ceil(rb_up_x/S["dx"]))
    ii_e_temp = int(np.ceil(rb_up_x/S["dx"]))
    jj_s_temp = -int(np.ceil(rb_up_y/S["dy"]))
    jj_e_temp = int(np.ceil(rb_up_y/S["dy"]))
    kk_s_temp = 0
    kk_e_temp = int(np.ceil(rb_up_z/S["dz"]))

    xx_temp = pos_atm_x + np.arange(ii_s_temp - S["FDn"], ii_e_temp + S["FDn"] + 1)*S["dx"]
    yy_temp = pos_atm_y + np.arange(jj_s_temp - S["FDn"], jj_e_temp + S["FDn"] + 1)*S["dy"]
    zz_temp = pos_atm_z + np.arange(kk_s_temp - S["FDn"], kk_e_temp + S["FDn"] + 1)*S["dz"]

    XX_3D_temp, YY_3D_temp, ZZ_3D_temp = np.meshgrid(xx_temp, yy_temp, zz_temp, indexing='ij')
    Nx = ii_e_temp - ii_s_temp + 1
    Ny = jj_e_temp - jj_s_temp + 1
    Nz = kk_e_temp - kk_s_temp + 1

    dd_temp = calculateDistance(XX_3D_temp, YY_3D_temp, ZZ_3D_temp,
                                pos_atm_x, pos_atm_y, pos_atm_z, S)

    W_temp = IntgWts(Nx, Ny, Nz, 1, 1, 1, xx_temp[S["FDn"]], S)
    # W_temp = W_temp.reshape(Nx, Ny, Nz)
    W_temp = W_temp.reshape(Nx, Ny, Nz, order='F')

    for ityp in range(S["n_typ"]):
        V_PS_temp = np.zeros_like(dd_temp)
        r_grid = S["Atm"][ityp]["r_grid_vloc"]
        Vloc = S["Atm"][ityp]["Vloc"]

        mask_out = dd_temp > r_grid[-1]
        V_PS_temp[mask_out] = -S["Atm"][ityp]["Z"]

        mask_in = ~mask_out
        if np.any(mask_in):
            interp_func = interp1d(r_grid, r_grid*Vloc, 'cubic')
            V_PS_temp[mask_in] = interp_func(dd_temp[mask_in])

        # divide elementwise, avoid near-core division by zero
        near_core = dd_temp < r_grid[1]
        V_PS_temp[~near_core] /= dd_temp[~near_core]
        V_PS_temp[near_core] = Vloc[0]

        II_temp = np.arange(S["FDn"], V_PS_temp.shape[0] - S["FDn"])
        JJ_temp = np.arange(S["FDn"], V_PS_temp.shape[1] - S["FDn"])
        KK_temp = np.arange(S["FDn"], V_PS_temp.shape[2] - S["FDn"])

        b_temp = pseudochargeDensity_atom(V_PS_temp, II_temp, JJ_temp, KK_temp, xx_temp[0], S)
        b_temp = -b_temp / (4*np.pi)

        rb_x = np.ceil(S["Atm"][ityp]["rc"]/S["dx"] - 1e-12)*S["dx"]
        rb_y = np.ceil(f_rby(S["Atm"][ityp]["rc"])/S["dy"] - 1e-12)*S["dy"]
        rb_z = np.ceil(S["Atm"][ityp]["rc"]/S["dz"] - 1e-12)*S["dz"]

        err_rb = 100.0
        count = 1

        #print(f'Finding rb for {S["Atm"][ityp]["typ"]} ...')
        while (err_rb > S["pseudocharge_tol"] and count <= 100 and
               rb_x <= rb_up_x and rb_y <= rb_up_y and rb_z <= rb_up_z):

            rb_x += S["dx"]
            rb_z += S["dz"]
            rb_y = f_rby(max(rb_x, rb_z))

            ii_rb = np.arange(-ii_s_temp + S["FDn"] - int(rb_x/S["dx"]),
                               -ii_s_temp + S["FDn"] + int(rb_x/S["dx"]) + 1)
            jj_rb = np.arange(-jj_s_temp + S["FDn"] - int(rb_y/S["dy"]),
                               -jj_s_temp + S["FDn"] + int(rb_y/S["dy"]) + 1)
            kk_rb = np.arange(S["FDn"], S["FDn"] + int(rb_z/S["dz"]) + 1)

            int_b = np.sum(W_temp[np.ix_(ii_rb - S["FDn"], jj_rb - S["FDn"], kk_rb - S["FDn"])] *
                           b_temp[np.ix_(ii_rb, jj_rb, kk_rb)])

            err_rb = abs(2*int_b + S["Atm"][ityp]["Z"])
            #print(f'rb = {{ {rb_x:.3f}, {rb_y:.3f}, {rb_z:.3f} }}, int_b = {2*int_b:.15f}, err_rb = {err_rb:.3e}')

            count += 1

        assert rb_x <= rb_up_x and rb_y <= rb_up_y and rb_z <= rb_up_z, \
            "Need to increase upper bound for rb!"

        S["Atm"][ityp]["rb_x"] = rb_x
        S["Atm"][ityp]["rb_y"] = rb_y
        S["Atm"][ityp]["rb_z"] = rb_z

        #print(f'rb = {{ {rb_x:.3f}, {rb_y:.3f}, {rb_z:.3f} }}')

    return S


import numpy as np
from scipy.interpolate import interp1d
import time

def calculate_b_guessRho_Eself(S):
    """
    Calculates the pseudocharge (& ref), self energy (& ref),
    electronic density guess, electrostatic energy correction
    and electrostatic potential correction.

    Ported from MATLAB to Python.
    """

    #print('\n Starting pseudocharge generation and self energy calculation...\n')
    t1 = time.time()

    # --- Initialization ---
    # CRITICAL FIX: Flatten W to ensure it is (N,) not (N,1)
    # This prevents the ValueError and incorrect broadcasting later.
    S['W'] = np.ravel(S['W'])

    # We assume S['b'], S['b_ref'], etc., are initialized as 1D arrays of size (Nx*Ny*Nz)
    # consistent with the SPARC structure.
    S['b'] = np.zeros(S['N'])
    S['b_ref'] = np.zeros(S['N'])
    S['rho_at'] = np.zeros(S['N'])


    S['Eself'] = 0.0
    S['Eself_ref'] = 0.0
    S['V_c'] = np.zeros(S['N'])

    # Create 3D views for easier slicing updates.
    # IMPORTANT: MATLAB indexing provided implies X is fastest (Fortran order).
    # modifying these views updates the underlying 1D arrays in S.
    b_view = S['b'].reshape((S['Nx'], S['Ny'], S['Nz']), order='F')
    b_ref_view = S['b_ref'].reshape((S['Nx'], S['Ny'], S['Nz']), order='F')
    rho_at_view = S['rho_at'].reshape((S['Nx'], S['Ny'], S['Nz']), order='F')
    Vc_view = S['V_c'].reshape((S['Nx'], S['Ny'], S['Nz']), order='F')

    # We also need W in 3D for the energy integration
    W_view = S['W'].reshape((S['Nx'], S['Ny'], S['Nz']), order='F')

    # Counters
    count_typ = 0 # 0-based index for atom types
    count_typ_atms = 1 # Keep 1-based to match original counter logic checks

    # --- Loop over all atoms ---
    for JJ_a in range(S['n_atm']):

        # Atom Position
        x0 = S['Atoms'][JJ_a, 0]
        y0 = S['Atoms'][JJ_a, 1]
        z0 = S['Atoms'][JJ_a, 2]

        # Get current atom type parameters
        # Accessing list S['Atm'][count_typ]
        current_atm = S['Atm'][count_typ]
        rb_x = current_atm['rb_x']
        rb_y = current_atm['rb_y']
        rb_z = current_atm['rb_z']

        # --- Image Calculation (Periodic Boundary Conditions) ---
        if S['BCx'] == 0:
            n_image_xl = int(np.floor((x0 + rb_x) / S['L1']))
            n_image_xr = int(np.floor((S['L1'] - x0 + rb_x - S['dx']) / S['L1']))
        else:
            n_image_xl = 0; n_image_xr = 0

        if S['BCy'] == 0:
            n_image_yl = int(np.floor((y0 + rb_y) / S['L2']))
            n_image_yr = int(np.floor((S['L2'] - y0 + rb_y - S['dy']) / S['L2']))
        else:
            n_image_yl = 0; n_image_yr = 0

        if S['BCz'] == 0:
            n_image_zl = int(np.floor((z0 + rb_z) / S['L3']))
            n_image_zr = int(np.floor((S['L3'] - z0 + rb_z - S['dz']) / S['L3']))
        else:
            n_image_zl = 0; n_image_zr = 0

        # Generate Image Coordinates
        xx_img = np.arange(-n_image_xl, n_image_xr + 1) * S['L1'] + x0
        yy_img = np.arange(-n_image_yl, n_image_yr + 1) * S['L2'] + y0
        zz_img = np.arange(-n_image_zl, n_image_zr + 1) * S['L3'] + z0

        XX_IMG_3D, YY_IMG_3D, ZZ_IMG_3D = np.meshgrid(xx_img, yy_img, zz_img, indexing='ij')

        # Flat list of image coordinates
        img_coords = np.column_stack((XX_IMG_3D.ravel(), YY_IMG_3D.ravel(), ZZ_IMG_3D.ravel()))

        # --- Loop over all images ---
        for img in img_coords:
            x0_i, y0_i, z0_i = img

            # ************************************************************************
            # * Calculate b, b_ref, Eself, Eself_ref and rho_at            *
            # ************************************************************************

            # Starting and ending indices of b-region (0-based)
            # MATLAB: ceil((x - rb)/dx) + 1  -> Python: ceil((x - rb)/dx) is the index *before*?
            # Let's map strict logic: index i corresponds to [i*dx, (i+1)*dx)
            # We want the range covering the sphere.

            ii_s = int(np.ceil((x0_i - rb_x) / S['dx']))
            ii_e = int(np.floor((x0_i + rb_x) / S['dx'])) # Python range is exclusive at end, so we might need +1 later

            jj_s = int(np.ceil((y0_i - rb_y) / S['dy']))
            jj_e = int(np.floor((y0_i + rb_y) / S['dy']))

            kk_s = int(np.ceil((z0_i - rb_z) / S['dz']))
            kk_e = int(np.floor((z0_i + rb_z) / S['dz']))

            # Boundary Checks (Dirichlet)
            isInside = (
                (S['BCx'] == 0 or (S['BCx'] == 1 and ii_s >= 0 and ii_e < S['Nx'])) and
                (S['BCy'] == 0 or (S['BCy'] == 1 and jj_s >= 0 and jj_e < S['Ny'])) and
                (S['BCz'] == 0 or (S['BCz'] == 1 and kk_s >= 0 and kk_e < S['Nz']))
            )

            if not isInside:
                 print(f" WARNING: Atom {JJ_a} too close to boundary for b calculation")

            # Clamp indices to domain (0 to N-1)
            ii_s = max(ii_s, 0)
            ii_e = min(ii_e, S['Nx'] - 1)
            jj_s = max(jj_s, 0)
            jj_e = min(jj_e, S['Ny'] - 1)
            kk_s = max(kk_s, 0)
            kk_e = min(kk_e, S['Nz'] - 1)

            # Generate Padded Local Grid for Finite Difference
            # Note: ii_e is the index of the last element. Python range needs +1.
            # Local grid generation:
            # We need FDn ghost nodes on left and right.
            # Grid coordinates corresponding to indices: ii_s-FDn ... ii_e+FDn

            idx_x_local = np.arange(ii_s - S['FDn'], ii_e + S['FDn'] + 1)
            idx_y_local = np.arange(jj_s - S['FDn'], jj_e + S['FDn'] + 1)
            idx_z_local = np.arange(kk_s - S['FDn'], kk_e + S['FDn'] + 1)

            xx = S['xin'] + idx_x_local * S['dx']
            yy = S['yin'] + idx_y_local * S['dy']
            zz = S['zin'] + idx_z_local * S['dz']

            XX_3D, YY_3D, ZZ_3D = np.meshgrid(xx, yy, zz, indexing='ij')

            # Find distances
            dd = calculateDistance(XX_3D, YY_3D, ZZ_3D, x0_i, y0_i, z0_i, S)

            # --- Pseudopotential (Interpolation) ---
            V_PS = np.zeros_like(dd)

            r_grid_vloc = current_atm['r_grid_vloc']
            Vloc = current_atm['Vloc']
            Z_val = current_atm['Z']

            IsLargeThanRmax = dd > r_grid_vloc[-1]
            V_PS[IsLargeThanRmax] = -Z_val

            mask_interp = ~IsLargeThanRmax
            if np.any(mask_interp):
                # Using cubic interpolation
                f_interp = interp1d(r_grid_vloc, r_grid_vloc * Vloc, kind='cubic',
                                    fill_value="extrapolate", bounds_error=False)
                V_PS[mask_interp] = f_interp(dd[mask_interp])

            # Element-wise division with protection
            # Avoid divide by zero
            valid_dd = dd > 1e-12
            V_PS[valid_dd] /= dd[valid_dd]

            # Handle small r (core)
            # MATLAB: V_PS(dd<r_grid_vloc(2)) = Vloc(1)
            near_core = dd < r_grid_vloc[1]
            V_PS[near_core] = Vloc[0]

            # --- Reference Potential ---
            rc_ref = S['rc_ref']
            V_PS_ref = np.zeros_like(dd)
            I_ref = dd < rc_ref

            # Outside reference radius
            mask_out_ref = ~I_ref
            if np.any(mask_out_ref):
                V_PS_ref[mask_out_ref] = -Z_val / dd[mask_out_ref]

            # Inside reference radius (Polynomial)
            if np.any(I_ref):
                d_in = dd[I_ref]
                term = (9 * d_in**7 - 30 * rc_ref * d_in**6
                        + 28 * rc_ref**2 * d_in**5 - 14 * (rc_ref**5) * d_in**2 + 12 * rc_ref**7)
                V_PS_ref[I_ref] = -Z_val * term / (5 * rc_ref**8)

            # --- Isolated Atom Electron Density ---
            r_grid_rho = current_atm['r_grid_rho']
            rho_guess = current_atm['rho_isolated_guess']

            # Use fill_value 0 for r > r_end to match MATLAB line: rho(dd > end) = 0
            f_rho = interp1d(r_grid_rho, rho_guess, kind='cubic',
                             bounds_error=False, fill_value=0.0)
            rho_isolated_atom = f_rho(dd)

            # --- Pseudocharge & Accumulation ---
            # Define Interior Indices relative to the local temporary grid
            # MATLAB: 1+FDn : size-FDn
            # Python: FDn : size-FDn (Upper bound exclusive in slice)

            # Note: bJ calculation needs the coordinates of the first point?
            # MATLAB passed xx(1).
            bJ = pseudochargeDensity_atom(V_PS,
                                          np.arange(S['FDn'], V_PS.shape[0]-S['FDn']),
                                          np.arange(S['FDn'], V_PS.shape[1]-S['FDn']),
                                          np.arange(S['FDn'], V_PS.shape[2]-S['FDn']),
                                          xx[0], S)

            bJ_ref = pseudochargeDensity_atom(V_PS_ref,
                                              np.arange(S['FDn'], V_PS_ref.shape[0]-S['FDn']),
                                              np.arange(S['FDn'], V_PS_ref.shape[1]-S['FDn']),
                                              np.arange(S['FDn'], V_PS_ref.shape[2]-S['FDn']),
                                              xx[0], S)

            # Slicing the interior of the computed blocks
            # The shape of bJ is effectively the interior size
            # We must map this to the global grid at [ii_s:ii_e, jj_s:jj_e, kk_s:kk_e]

            # Extraction slices for local arrays (removing ghost nodes)
            sl_local = (slice(S['FDn'], -S['FDn']),
                        slice(S['FDn'], -S['FDn']),
                        slice(S['FDn'], -S['FDn']))

            bJ_int = bJ[sl_local]
            bJ_ref_int = bJ_ref[sl_local]
            rho_add = rho_isolated_atom[sl_local]

            # Corresponding Potentials for Energy Integration
            V_PS_int = V_PS[sl_local]
            V_PS_ref_int = V_PS_ref[sl_local]

            # --- Update Global Arrays ---
            # Using 3D views (Fortran Order) for direct block update
            # Indices: ii_s to ii_e inclusive. Python slice: ii_s : ii_e + 1

            target_slice = (slice(ii_s, ii_e + 1),
                            slice(jj_s, jj_e + 1),
                            slice(kk_s, kk_e + 1))

            b_view[target_slice] += bJ_int
            b_ref_view[target_slice] += bJ_ref_int
            rho_at_view[target_slice] += rho_add

            # V_c update
            Vc_view[target_slice] += (V_PS_ref_int - V_PS_int)

            # --- Energy Integration ---
            # Access weights for this block
            W_local = W_view[target_slice]

            S['Eself'] += 0.5 * np.sum(bJ_int * V_PS_int * W_local)
            S['Eself_ref'] += 0.5 * np.sum(bJ_ref_int * V_PS_ref_int * W_local)

        # --- Update Atom Type Counters ---
        if count_typ_atms == current_atm['n_atm_typ']:
            count_typ_atms = 1
            count_typ += 1
        else:
            count_typ_atms += 1

    # --- Final Scaling and Charge Calculations ---

    # Scaling factor
    scale_fac = -1.0 / (4.0 * np.pi)

    S['b'] *= scale_fac
    S['Eself'] *= scale_fac
    S['b_ref'] *= scale_fac
    S['Eself_ref'] *= scale_fac

    # Charge Checks
    S['PosCharge'] = np.abs(np.dot(S['W'], S['b']))
    S['NegCharge'] = -S['PosCharge'] + S['NetCharge']

    print(f" Integration b = {np.abs(np.dot(S['W'], S['b'])):.12f}\n")
    print(f" Integration b_ref = {np.abs(np.dot(S['W'], S['b_ref'])):.12f}\n")

    # --- Designate rho_at ---

    # Scaling rho_at to match NegCharge
    integral_rho = np.dot(S['W'], S['rho_at'])
    #print(f'Integral rho before scaling: {integral_rho}')
    if integral_rho != 0:
        rho_scal = np.abs(S['NegCharge'] / integral_rho)
        S['rho_at'] *= rho_scal


    #print(' ****************************************')
    #print(f" * Eself_ref = {S['Eself_ref']:.6f}        *")
    #print(' ****************************************')

    # --- Calculate E_corr ---
    # Formula: 0.5 * sum((b_ref + b) * V_c * W) + Eself - Eself_ref

    term1 = 0.5 * np.sum((S['b_ref'] + S['b']) * S['V_c'] * S['W'])
    S['E_corr'] = term1 + S['Eself'] - S['Eself_ref']
    #print(f" * E_corr = {S['E_corr']:.6f}        *")

    print(f" Done. ({time.time() - t1:.4f} s)\n")

    # NLCC part skipped as requested.

    return S

import sys
from numba import jit

# Initialize dictionary
formulas = []
rhos = []

def execute(fname):
    S = {}
    # Mesh in Angstorms
    S['mesh_spacing'] = 0.2
    # Finite difference order = 2*FDn
    S['FDn'] = 6
    # Filepath
    S['inputfile_path'] = './'

    # Periodic boundary conditions (0), Dirichlet condition change to 1
    S['BCx'] = 0
    S['BCy'] = 0
    S['BCz'] = 0

    # Store tolerance
    S['temp_tol'] = 1e-14

    # Read and store info from the POSCARS
    S, formula = read_and_store_info(S,fname)

    # Origin
    S['xin'] = 0.0
    S['yin'] = 0.0
    S['zin'] = 0.0

    # Reference cutoff
    S['rc_ref'] = 0.5

    # Calculate pseudocharge radius
    S['pseudocharge_tol'] = 1e-8
    S = Calculate_rb(S)

    # Calculate guess rho
    S['NetCharge'] = 0
    S = calculate_b_guessRho_Eself(S)

    # The atomic density guess is flat array of size S['N']
    # -- reshape in (S['Nx'], S['Ny'], S['Nz']) format if needed
    # If required in number_of_electron/angstorm^3 then multiply with (1/Bohr)^3 element-wise
    rho = S['rho_at']/(Bohr**3)

    rho3d = rho.reshape((S['Nx'], S['Ny'], S['Nz']), order='F')

    return rho3d, formula

# Filename of structures (POSCAR only)
#fname = './RHEA4_SQSPOSCAR/SQSPOSCAR-00001'

c=0
#dir = os.path.join(HOME, 'RHEA4_rest')

for fname in os.listdir(os.path.join(HOMED, 'RHEA4latt')): #RHEA4POSCAR
    #plotvista(rho3d)
    if fname[-7:]!='.POSCAR':
        try:
            os.rename(os.path.join(HOMED, 'RHEA4latt', fname), os.path.join(HOMED, 'RHEA4latt',f"{fname}.POSCAR"))
            fname = fname+'.POSCAR'
        except:
            continue
    #print(fname)
    c+=1
    save_path= os.path.join(HOMED,'RelaxLatt2PS',f"{fname[:-7]}.h5")
    if os.path.isfile(save_path):
        print(f"Formula: {fname[:-7]} | Sample {c} | Meow Meow Meow")
        continue
    else:
        print(f"Processing Formula: {fname[:-7]} | Sample {c}")
    #print(f"Processing {fname}")
    rho3d, formula = execute(os.path.join(HOMED, 'RHEA4latt',fname))
    tpt = tptstats(rho3d)
    #slicedplot(tpt)
    #print(f"Formula: {formula} | Sample {c}")

    try:
        if not os.path.isfile(save_path):
          print(f"Saving to {save_path}")
          #h5f.close()
          h5f = h5py.File(save_path, 'w')
          h5f.create_dataset('tpt', data=rho3d)
          h5f.create_dataset('formula', data=[formula])
          h5f.close()
          formulas.append(formula)
          rhos.append(rho3d.flatten())
          print(f"{formula} : {np.shape(rho3d)}")
          del rho3d
        else:
          continue
          with h5py.File(save_path, 'r') as f:
              rho3d = f['tpt']
              formula = f['formula'][0]
              formulas.append(formula)
              rhos.append(rho3d.flatten())
              print(f"{formula} : {np.shape(rho3d)}")
              f.close()
              continue
    except:
          h5f.close()
    #"""

print(len(os.listdir(os.path.join(HOMED,'RelaxLatt2PS'))))
# integral_rho = np.dot(S['W'], rho)
# print(f'Integral rho after scaling: {integral_rho}')

