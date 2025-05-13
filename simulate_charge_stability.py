#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Schrödinger-Poisson simulator for a semiconductor quantum dot device.
Modified to simulate charge stability diagrams by sweeping two gate voltages.
"""

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os  # Added for creating output directory
import numpy.fft as fft  # Added for spectral methods
from hilbertcurve.hilbertcurve import HilbertCurve # Added for Hilbert curve warm start

# --- Physical Constants ---
hbar = const.hbar
m_e = const.m_e
e = const.e
epsilon_0 = const.epsilon_0

# --- Material Parameters (e.g., GaAs) ---
m_eff = 0.067 * m_e  # Effective mass
eps_r = 12.9  # Relative permittivity
epsilon = eps_r * epsilon_0

# --- Simulation Grid ---
# Reduced grid for faster stability diagram calculation
Lx = 150e-9  # Length of the simulation domain in x (m)
Ly = 100e-9  # Length of the simulation domain in y (m)
Nx = 45  # Number of grid points in x (reduced)
Ny = 30  # Number of grid points in y (reduced)
N_total = Nx * Ny

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(
    x, y, indexing="ij"
)  # Important: 'ij' indexing matches matrix layout

# --- Spectral Method Setup ---
# Define k-space grid for spectral methods (assuming periodic boundary conditions)
kx = 2 * np.pi * fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * fft.fftfreq(Ny, d=dy)
Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
K_sq = Kx**2 + Ky**2
# Avoid division by zero at K=0 (DC component).
# For periodic BCs, the DC component of the potential is arbitrary; setting it to 0 is common.
# The inverse Laplacian of the DC component is undefined, so we handle it separately.
# Replace K_sq[0,0] with a small non-zero value or handle the DC term explicitly in the solver.
# Handling explicitly in the solver is cleaner.


# --- Device Parameters ---
def get_external_potential(X, Y, voltages):
    """
    Calculates the 2D external potential profile based on gate voltages.
    Uses 2D Gaussian profiles for gate influence.
    """
    potential = np.zeros_like(X)  # Potential in Joules

    # Define gate parameters (centers and widths in meters)
    # Example: A double dot defined by top gates
    gate_std_dev_x = 20e-9
    gate_std_dev_y = 20e-9  # Can be different for x and y

    # Gate positions (example) - Same as simulate_2d_dot
    p1_center = (Lx * 0.35, Ly * 0.5)
    p2_center = (Lx * 0.65, Ly * 0.5)
    b1_center = (Lx * 0.15, Ly * 0.5)  # Left barrier
    b2_center = (Lx * 0.50, Ly * 0.5)  # Center barrier
    b3_center = (Lx * 0.85, Ly * 0.5)  # Right barrier

    # Helper function for 2D Gaussian potential shape
    def gaussian_potential_2d(center_x, center_y, std_dev_x, std_dev_y, amplitude):
        return amplitude * np.exp(
            -(
                (X - center_x) ** 2 / (2 * std_dev_x**2)
                + (Y - center_y) ** 2 / (2 * std_dev_y**2)
            )
        )

    # Add potential contributions from each gate
    potential += gaussian_potential_2d(
        p1_center[0],
        p1_center[1],
        gate_std_dev_x,
        gate_std_dev_y,
        voltages.get("P1", 0.0) * e,
    )
    potential += gaussian_potential_2d(
        p2_center[0],
        p2_center[1],
        gate_std_dev_x,
        gate_std_dev_y,
        voltages.get("P2", 0.0) * e,
    )
    potential += gaussian_potential_2d(
        b1_center[0],
        b1_center[1],
        gate_std_dev_x,
        gate_std_dev_y,
        voltages.get("B1", 0.0) * e,
    )
    potential += gaussian_potential_2d(
        b2_center[0],
        b2_center[1],
        gate_std_dev_x,
        gate_std_dev_y,
        voltages.get("B2", 0.0) * e,
    )
    potential += gaussian_potential_2d(
        b3_center[0],
        b3_center[1],
        gate_std_dev_x,
        gate_std_dev_y,
        voltages.get("B3", 0.0) * e,
    )

    return potential  # Potential in Joules


# --- Schrödinger Solver (2D) ---
def solve_schrodinger_2d(potential_2d):
    """
        Solves the 2D time-independent Schrödinger equation.
        Returns eigenvalues (energies) and eigenvectors (wavefunctions reshaped to
    2D).
    """
    potential_flat = potential_2d.flatten(order="C")  # Flatten consistently

    # Construct the 2D Hamiltonian using sparse matrix format
    diag = hbar**2 / (m_eff * dx**2) + hbar**2 / (m_eff * dy**2) + potential_flat
    offdiag_x = -(hbar**2) / (2 * m_eff * dx**2) * np.ones(N_total)
    offdiag_y = -(hbar**2) / (2 * m_eff * dy**2) * np.ones(N_total)

    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0

    H = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    try:
        num_eigenstates = 10  # Adjust as needed
        eigenvalues, eigenvectors_flat = spla.eigsh(H, k=num_eigenstates, which="SM")
    except Exception as e:
        print(f"Eigenvalue solver failed: {e}")
        return np.array([]), np.empty((Nx, Ny, 0))

    eigenvectors = np.zeros((Nx, Ny, num_eigenstates))
    for i in range(num_eigenstates):
        psi_flat = eigenvectors_flat[:, i]
        norm = np.sqrt(np.sum(np.abs(psi_flat) ** 2) * dx * dy)
        if norm > 1e-12: # Avoid division by zero for zero vectors
            eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")
        else:
            eigenvectors[:, :, i] = psi_flat.reshape((Nx, Ny), order="C")


    return eigenvalues, eigenvectors


# --- Charge Density Calculation (2D) ---
# (Using zero temperature version from simulate_2d_dot.py)
def calculate_charge_density_2d(eigenvalues, eigenvectors_2d, fermi_level):
    """
    Calculates the 2D electron charge density (C/m^2).
    Assumes zero temperature.
    """
    density_2d = np.zeros((Nx, Ny))  # electrons/m^2
    num_states = eigenvectors_2d.shape[2]

    for i in range(num_states):
        if eigenvalues[i] < fermi_level:
            density_2d += 2 * np.abs(eigenvectors_2d[:, :, i]) ** 2  # Factor 2 for spin
        else:
            break

    charge_density_2d = -e * density_2d  # Coulombs per square meter (C/m^2)
    return charge_density_2d


# --- Total Electron Number Calculation ---
def calculate_total_electrons(charge_density_2d):
    """Calculates the total number of electrons by integrating the charge density."""
    # Ensure charge_density_2d is a numpy array
    charge_density_2d = np.asarray(charge_density_2d)

    # Integrate density (electrons/m^2) over area (dx*dy)
    total_charge = np.sum(charge_density_2d * dx * dy)
    total_electrons = total_charge / (-e)
    return total_electrons


# --- Poisson Solver (2D) ---
# (Identical to simulate_2d_dot.py - no changes needed here)
def solve_poisson_2d(charge_density_2d):
    """
        Solves the 2D Poisson equation: laplacian(phi) = -rho / epsilon
        Returns the 2D electrostatic potential phi (Volts).
        Uses finite differences and assumes Dirichlet boundary conditions (phi=0 on
    boundary).
    """
    rho_flat = charge_density_2d.flatten(order="C")

    diag = (-2 / dx**2 - 2 / dy**2) * np.ones(N_total)
    offdiag_x = (1 / dx**2) * np.ones(N_total)
    offdiag_y = (1 / dy**2) * np.ones(N_total)

    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0

    A = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")
    b = -rho_flat / epsilon

    A = A.tolil()
    b_modified = b.copy()

    boundary_indices = []
    boundary_indices.extend(range(0, N_total, Ny))
    boundary_indices.extend(range(Ny - 1, N_total, Ny))
    boundary_indices.extend(range(1, Ny - 1))
    boundary_indices.extend(range(N_total - Ny + 1, N_total - 1))
    boundary_indices = sorted(list(set(boundary_indices)))

    for idx in boundary_indices:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]
        b_modified[idx] = 0.0

    A = A.tocsc()

    try:
        phi_flat = spla.spsolve(A, b_modified)
    except spla.MatrixRankWarning:
        print("Warning: Poisson matrix singular. Using iterative solver.")
        phi_flat, info = spla.gmres(A, b_modified, tol=1e-8, maxiter=2 * N_total)
        if info != 0:
            print(f"Poisson solver (GMRES) did not converge (info={info}).")
            phi_flat = np.zeros_like(rho_flat)
    except Exception as e:
        print(f"Poisson solver failed: {e}")
        phi_flat = np.zeros_like(rho_flat)

    phi_2d = phi_flat.reshape((Nx, Ny), order="C")
    return phi_2d  # Electrostatic potential in Volts


# --- Poisson Solver (2D) - Spectral Method ---
def solve_poisson_2d_spectral(charge_density_2d):
    """
    Solves the 2D Poisson equation: laplacian(phi) = -rho / epsilon
    Returns the 2D electrostatic potential phi (Volts).
    Uses spectral methods (FFT) and assumes periodic boundary conditions.
    """
    # 1. FFT of charge density
    rho_k = fft.fft2(charge_density_2d)

    # 2. Solve in Fourier space: phi_k = -rho_k / (epsilon * K_sq)
    # Handle the DC component (k=0,0) separately.
    # For periodic BCs, the average potential is arbitrary.
    # Setting phi_k[0,0] = 0 corresponds to zero average potential.
    # Create a copy of K_sq to avoid modifying the global variable
    K_sq_solver = K_sq.copy()
    K_sq_solver[0, 0] = 1.0  # Set to 1 to avoid division by zero for the DC term

    phi_k = -rho_k / (epsilon * K_sq_solver)
    phi_k[0, 0] = 0.0  # Explicitly set DC component to zero

    # 3. Inverse FFT to get potential in real space
    phi_2d = fft.ifft2(phi_k).real  # Take real part as potential is real

    return phi_2d  # Electrostatic potential in Volts


# --- Hilbert Curve Ordering ---
def get_hilbert_order(nx, ny):
    """
    Generates a list of (i, j) indices in Hilbert curve order for an nx x ny grid.
    """
    # Determine the Hilbert curve level p such that 2^p >= max(nx, ny)
    p = int(np.ceil(np.log2(max(nx, ny))))
    n_dims = 2
    hilbert_curve = HilbertCurve(p, n_dims)

    # Generate all grid points (indices)
    points = [(i, j) for i in range(nx) for j in range(ny)]

    # Calculate Hilbert distances for each point
    distances = hilbert_curve.distances_from_points(points)

    # Sort points based on Hilbert distances
    hilbert_ordered_indices = [point for _, point in sorted(zip(distances, points))]

    return hilbert_ordered_indices


# --- Self-Consistent Iteration (2D) ---
# (Modified slightly for stability diagram context)
def self_consistent_solver_2d(
    voltages,
    fermi_level,
    max_iter=30,
    tol=1e-4,
    mixing=0.1,
    verbose=False,
    initial_potential_V=None,  # Keep warm start parameter
    poisson_solver_type="finite_difference",  # Add solver type option
):
    """
    Performs the self-consistent 2D Schrödinger-Poisson calculation.
    Allows choosing the Poisson solver ('finite_difference' or 'spectral').
    Returns the final charge density and the converged electrostatic potential.
    """
    if verbose:
        print(f"Running SC calculation for voltages: {voltages}")
    start_time_sc = time.time()

    electrostatic_potential_V = np.zeros((Nx, Ny))
    external_potential_J = get_external_potential(X, Y, voltages)

    final_charge_density = None  # Initialize in case of early exit
    # Initialize converged_potential_V with the initial guess or zero potential
    converged_potential_V = electrostatic_potential_V.copy()


    for i in range(max_iter):
        total_potential_J = external_potential_J - e * electrostatic_potential_V
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(total_potential_J)

        if not eigenvalues.size: # Original check
            print("Error in Schrödinger solver during SC iteration. Aborting.")
            return None, None  # Indicate failure by returning None for both values

        new_charge_density = calculate_charge_density_2d(
            eigenvalues, eigenvectors_2d, fermi_level
        )
        final_charge_density = new_charge_density  # Store the latest density

        # 4. Solve Poisson equation using the selected solver
        if poisson_solver_type == "finite_difference":
            new_electrostatic_potential_V = solve_poisson_2d(new_charge_density)
        elif poisson_solver_type == "spectral":
            # Note: Spectral solver assumes periodic boundary conditions,
            # which may differ from the desired physics (Dirichlet).
            new_electrostatic_potential_V = solve_poisson_2d_spectral(
                new_charge_density
            )
        else:
            raise ValueError(f"Unknown poisson_solver_type: {poisson_solver_type}")

        # 5. Check for convergence (using norm of potential difference)
        potential_diff_norm = np.linalg.norm(
            new_electrostatic_potential_V - electrostatic_potential_V
        ) * np.sqrt(dx * dy)

        if verbose and (i % 5 == 0 or i == max_iter - 1):  # Print progress less often
            print(
                f"  SC Iter {i + 1}/{max_iter}, Potential diff norm: {potential_diff_norm:.3e}"
            )

        if potential_diff_norm < tol:
            if verbose:
                print(f"  Converged after {i + 1} iterations.")
            electrostatic_potential_V = new_electrostatic_potential_V
            break

        electrostatic_potential_V = electrostatic_potential_V + mixing * (
            new_electrostatic_potential_V - electrostatic_potential_V
        )
    else:
        print(
            f"Warning: SC loop did not converge after {max_iter} iterations for {voltages}."
        )

    end_time_sc = time.time()
    if verbose:
        print(f"  SC loop time: {end_time_sc - start_time_sc:.2f} seconds")

    # Assign the final electrostatic potential before returning
    converged_potential_V = electrostatic_potential_V

    # Return the last calculated charge density and the final electrostatic potential
    return final_charge_density, converged_potential_V


# --- Main Execution ---
if __name__ == "__main__":
    # Define fixed applied voltages (Volts) - Barriers etc.
    base_voltages = {
        # "P1": -0.20, # Will be swept
        # "P2": -0.20, # Will be swept
        "B1": 0.15,
        "B2": 0.20,
        "B3": 0.15,
    }

    # --- Stability Diagram Sweep Parameters ---
    gate1_name = "P1"
    gate2_name = "P2"
    # Define voltage ranges for the sweep (adjust for desired charge transitions)
    gate1_voltages = np.linspace(-0.1, 0.0, 121)  # Coarser sweep for speed
    gate2_voltages = np.linspace(-0.1, 0.0, 121)  # Coarser sweep for speed
    num_v1 = len(gate1_voltages)
    num_v2 = len(gate2_voltages)

    # Array to store results (total electron number)
    total_electron_map = np.full((num_v1, num_v2), np.nan)  # Use NaN for failed points

    # Define Fermi level (relative to minimum external potential of a reference state)
    # Calculate based on a typical operating point, keep constant during sweep.
    ref_voltages = base_voltages.copy()
    ref_voltages[gate1_name] = np.mean(gate1_voltages)  # Use mid-range voltage
    ref_voltages[gate2_name] = np.mean(gate2_voltages)
    initial_ext_pot_J = get_external_potential(X, Y, ref_voltages)
    # Set Fermi level slightly above the potential minimum of the reference configuration
    # to ensure some states are occupied within the sweep range.
    fermi_level_J = (
        np.min(initial_ext_pot_J) + 0.05 * e
    )  # Example: 50 meV above min potential

    print("\n--- Starting Charge Stability Diagram Simulation ---")
    print(
        f"Sweeping {gate1_name} from {gate1_voltages[0]:.3f} V to {gate1_voltages[-1]:.3f} V ({num_v1} points)"
    )
    print(
        f"Sweeping {gate2_name} from {gate2_voltages[0]:.3f} V to {gate2_voltages[-1]:.3f} V ({num_v2} points)"
    )
    print(f"Grid size: Nx={Nx}, Ny={Ny}")
    print(f"Fermi Level: {fermi_level_J / e:.4f} eV")
    print("-" * 50)

    output_dir = "stability_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in '{output_dir}'")

    sweep_start_time = time.time()
    total_points = num_v1 * num_v2
    completed_points = 0

    # --- Choose Sweep Strategy ---
    # Options: 'row_by_row', 'hilbert'
    sweep_strategy = "hilbert" # Change this to 'row_by_row' for the original behavior

    # Initialize warm start potential variable for Hilbert sweep
    potential_from_previous_point = None

    # Initialize a map to store converged potentials for grid points
    # Use a list of lists to store numpy arrays (potentials) or None
    converged_potentials_map = [[None for _ in range(num_v2)] for _ in range(num_v1)]

    # --- Loop through sweep voltages based on strategy ---
    if sweep_strategy == "hilbert":
        print("Using Hilbert curve sweep strategy...")
        hilbert_indices = get_hilbert_order(num_v1, num_v2)
        total_points = len(hilbert_indices)

        for point_index, (i, j) in enumerate(hilbert_indices):
            v1 = gate1_voltages[i]
            v2 = gate2_voltages[j]

            point_start_time = time.time()
            current_voltages = base_voltages.copy()
            current_voltages[gate1_name] = v1
            current_voltages[gate2_name] = v2

            print(
                f"Running Hilbert point {point_index + 1}/{total_points} (Grid: {i + 1},{j + 1}): "
                f"{gate1_name}={v1:.3f}V, {gate2_name}={v2:.3f}V"
            )

            # Determine warm start potential
            warm_start_potential = potential_from_previous_point # Default to previous Hilbert point

            # Check grid neighbor (i-1, j) if available and successful
            # This adds consideration of a direct grid neighbor
            if i > 0 and converged_potentials_map[i-1][j] is not None:
                # print(f"  Using grid neighbor ({i-1},{j}) as warm start.") # Optional: uncomment for debugging
                warm_start_potential = converged_potentials_map[i-1][j]
            # Could add check for (i, j-1) as well, but prioritizing one is simpler for now.

            # Run the self-consistent solver
            final_charge_density, converged_potential_V = self_consistent_solver_2d(
                current_voltages,
                fermi_level_J,
                max_iter=20,
                tol=5e-4,
                mixing=0.1,
                verbose=False,
                initial_potential_V=warm_start_potential, # Use the determined warm start
                poisson_solver_type="finite_difference",
            )

            if final_charge_density is not None:
                total_electrons = calculate_total_electrons(final_charge_density)
                total_electron_map[i, j] = total_electrons # Store using original grid indices
                converged_potentials_map[i][j] = converged_potential_V # Store converged potential
                print(f"  -> Total Electrons: {total_electrons:.3f}")
                potential_from_previous_point = converged_potential_V # Update for next Hilbert point
            else:
                print(
                    f"  -> Simulation failed for point ({i + 1},{j + 1}). Storing NaN."
                )
                total_electron_map[i, j] = np.nan
                converged_potentials_map[i][j] = None # Store None for failed points
                # Don't update potential_from_previous_point if failed

            completed_points += 1
            point_end_time = time.time()
            elapsed_time = point_end_time - sweep_start_time
            time_per_point = elapsed_time / completed_points
            estimated_remaining = (total_points - completed_points) * time_per_point
            print(
                f"  Point time: {point_end_time - point_start_time:.2f}s. Est. remaining: {estimated_remaining:.1f}s"
            )

    elif sweep_strategy == "row_by_row":
        print("Using row-by-row sweep strategy...")
        potential_from_previous_point_in_row = None # Specific to row strategy

        for i, v1 in enumerate(gate1_voltages):
            potential_from_previous_point_in_row = None # Reset for each new row

            for j, v2 in enumerate(gate2_voltages):
                point_start_time = time.time()
                current_voltages = base_voltages.copy()
                current_voltages[gate1_name] = v1
                current_voltages[gate2_name] = v2

                print(
                    f"Running point ({i + 1}/{num_v1}, {j + 1}/{num_v2}): {gate1_name}={v1:.3f}V, {gate2_name}={v2:.3f}V"
                )

                # Run the self-consistent solver with warm start from previous point in the row
                final_charge_density, converged_potential_V = self_consistent_solver_2d(
                    current_voltages,
                    fermi_level_J,
                    max_iter=20,
                    tol=5e-4,
                    mixing=0.1,
                    verbose=False,
                    initial_potential_V=potential_from_previous_point_in_row, # Use potential from previous point in row
                    poisson_solver_type="finite_difference",
                )

                if final_charge_density is not None:
                    total_electrons = calculate_total_electrons(final_charge_density)
                    total_electron_map[i, j] = total_electrons
                    print(f"  -> Total Electrons: {total_electrons:.3f}")
                    potential_from_previous_point_in_row = converged_potential_V # Update for next point in row
                else:
                    print(
                        f"  -> Simulation failed for point ({i + 1},{j + 1}). Storing NaN."
                    )
                    total_electron_map[i, j] = np.nan
                    # Don't update potential_from_previous_point_in_row if failed

                completed_points += 1
                point_end_time = time.time()
                elapsed_time = point_end_time - sweep_start_time
                time_per_point = elapsed_time / completed_points
                estimated_remaining = (total_points - completed_points) * time_per_point
                print(
                    f"  Point time: {point_end_time - point_start_time:.2f}s. Est. remaining: {estimated_remaining:.1f}s"
                )
    else:
        raise ValueError(f"Unknown sweep_strategy: {sweep_strategy}")

    sweep_end_time = time.time()
    print("-" * 50)
    print(
        f"Stability diagram simulation finished in {sweep_end_time - sweep_start_time:.2f} seconds."
    )

    # Save the raw data
    data_filename = os.path.join(
        output_dir, f"stability_data_{gate1_name}_{gate2_name}.npz"
    )
    np.savez(
        data_filename,
        gate1_voltages=gate1_voltages,
        gate2_voltages=gate2_voltages,
        total_electron_map=total_electron_map,
        base_voltages=base_voltages,
        fermi_level_J=fermi_level_J,
    )
    print(f"Raw data saved to {data_filename}")

    # --- Plotting Charge Stability Diagram ---
    print("Plotting results...")
    plt.figure(figsize=(8, 7))

    # Use pcolormesh for better handling of grid boundaries
    # Need meshgrid for pcolormesh coordinates
    V1, V2 = np.meshgrid(gate1_voltages, gate2_voltages, indexing="ij")
    # Transpose map because pcolormesh expects Z[j, i] for X[i], Y[j] if shading='flat' or similar issues
    plot_data = total_electron_map  # Keep as [i, j] corresponding to V1[i], V2[j]

    # Handle potential NaN values if desired (e.g., set to a specific color)
    plot_data_masked = np.ma.masked_invalid(plot_data)  # Mask NaN values

    cmap = plt.cm.viridis
    cmap.set_bad(color="grey")  # Color for NaN values

    # Determine reasonable color limits, focusing on integer transitions
    min_electrons = np.nanmin(plot_data)
    max_electrons = np.nanmax(plot_data)
    cmin = np.floor(min_electrons) - 0.5 if not np.isnan(min_electrons) else -0.5
    cmax = np.ceil(max_electrons) + 0.5 if not np.isnan(max_electrons) else 1.5

    pcm = plt.pcolormesh(
        V1, V2, plot_data_masked, cmap=cmap, shading="auto", vmin=cmin, vmax=cmax
    )
    plt.colorbar(pcm, label="Total Number of Electrons")

    plt.xlabel(f"Gate Voltage {gate1_name} (V)")
    plt.ylabel(f"Gate Voltage {gate2_name} (V)")
    plt.title(f"Charge Stability Diagram (Total Electrons)\nFixed: {base_voltages}")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.axis("tight")  # Adjust axes to fit data

    plot_filename = os.path.join(
        output_dir, f"charge_stability_{gate1_name}_{gate2_name}.png"
    )
    plt.savefig(plot_filename)
    print(f"Charge stability plot saved to {plot_filename}")
    plt.show()  # Force plot to display

    # Optional: Plot rounded electron numbers to emphasize plateaus
    try:
        plt.figure(figsize=(8, 7))
        rounded_electrons = np.round(plot_data)
        rounded_electrons_masked = np.ma.masked_invalid(rounded_electrons)
        # Use discrete colormap or levels for integer steps
        n_levels = int(np.nanmax(rounded_electrons) - np.nanmin(rounded_electrons)) + 1
        levels = (
            np.arange(
                np.floor(np.nanmin(rounded_electrons)),
                np.ceil(np.nanmax(rounded_electrons)) + 1,
            )
            - 0.5
        )
        # Use plt.colormaps.get_cmap instead of plt.cm.get_cmap
        cmap_discrete = plt.colormaps.get_cmap(
            "viridis", n_levels if n_levels > 0 else 1
        )
        cmap_discrete.set_bad(color="grey")

        pcm_rounded = plt.pcolormesh(
            V1,
            V2,
            rounded_electrons_masked,
            cmap=cmap_discrete,
            shading="auto",
            vmin=levels.min(),
            vmax=levels.max(),
        )
        cbar = plt.colorbar(
            pcm_rounded,
            label="Rounded Total Electrons",
            ticks=np.round(levels[:-1] + 0.5),
        )  # Ticks at integers
        # cbar.set_ticks(np.arange(int(np.nanmin(rounded_electrons)), int(np.nanmax(rounded_electrons)) + 1))

        plt.xlabel(f"Gate Voltage {gate1_name} (V)")
        plt.ylabel(f"Gate Voltage {gate2_name} (V)")
        plt.title(
            f"Charge Stability Diagram (Rounded Total Electrons)\nFixed: {base_voltages}"
        )
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.axis("tight")

        plot_filename_rounded = os.path.join(
            output_dir, f"charge_stability_rounded_{gate1_name}_{gate2_name}.png"
        )
        plt.savefig(plot_filename_rounded)
        plt.show()  # Force plot to display
        print(f"Rounded charge stability plot saved to {plot_filename_rounded}")

    except Exception as e:
        print(f"Error during rounded plot generation: {e}")

    print("Done.")
