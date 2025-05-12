#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Schrödinger-Poisson simulator for simulating Coulomb diamonds in a
semiconductor quantum dot device.
"""

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

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
Lx = 150e-9  # Length of the simulation domain in x (m)
Ly = 100e-9  # Length of the simulation domain in y (m)
Nx = 75  # Number of grid points in x (adjust for performance)
Ny = 50  # Number of grid points in y (adjust for performance)
N_total = Nx * Ny

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(
    x, y, indexing="ij"
)  # Important: 'ij' indexing matches matrix layout


# --- Device Parameters ---
def get_external_potential(X, Y, voltages):
    """
    Calculates the 2D external potential profile based on gate voltages.
    Uses 2D Gaussian profiles for gate influence.
    """
    potential = np.zeros_like(X)  # Potential in Joules

    # Define gate parameters (centers and widths in meters)
    # Example: A double dot defined by top gates
    gate_std_dev_x = 10e-9  # Reduced gate size
    gate_std_dev_y = 10e-9  # Reduced gate size

    # Gate positions (example)
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
    Returns eigenvalues (energies) and eigenvectors (wavefunctions reshaped to 2D).
    """
    potential_flat = potential_2d.flatten(order="C")  # Flatten consistently

    # Construct the 2D Hamiltonian using sparse matrix format
    # H = -hbar^2/(2*m_eff) * (d^2/dx^2 + d^2/dy^2) + V(x,y)
    # Finite difference approximation (5-point stencil)

    # Diagonal terms
    diag = hbar**2 / (m_eff * dx**2) + hbar**2 / (m_eff * dy**2) + potential_flat
    # Off-diagonal terms for x-derivatives
    offdiag_x = -(hbar**2) / (2 * m_eff * dx**2) * np.ones(N_total)
    # Off-diagonal terms for y-derivatives
    offdiag_y = -(hbar**2) / (2 * m_eff * dy**2) * np.ones(N_total)

    # Create sparse matrix diagonals
    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    # Define offsets for diagonals
    # 0: main diagonal
    # -1, 1: x-neighbors (careful with boundaries)
    # -Nx, Nx: y-neighbors
    offsets = [0, -1, 1, -Nx, Nx]

    # Adjust off-diagonals at boundaries (where stencil wraps around)
    # Remove connections between y=0 and y=Ny-1 (for offset -1)
    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    # Remove connections between y=Ny-1 and y=0 (for offset 1)
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0  # Corrected index

    H = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    # Find the lowest few eigenvalues and eigenvectors
    try:
        num_eigenstates = 10  # Adjust as needed
        # Use which='SM' (Smallest Magnitude)
        eigenvalues, eigenvectors_flat = spla.eigsh(H, k=num_eigenstates, which="SM")
    except Exception as e:
        print(f"Eigenvalue solver failed: {e}")
        return np.array([]), np.empty((Nx, Ny, 0))

    # Normalize and reshape eigenvectors
    eigenvectors = np.zeros((Nx, Ny, num_eigenstates))
    for i in range(num_eigenstates):
        psi_flat = eigenvectors_flat[:, i]
        norm = np.sqrt(np.sum(np.abs(psi_flat) ** 2) * dx * dy)
        eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")

    return eigenvalues, eigenvectors  # Eigenvalues in J, eigenvectors are 2D arrays


# --- Charge Density Calculation (2D) ---
def calculate_charge_density_2d(eigenvalues, eigenvectors_2d, fermi_level):
    """
    Calculates the 2D electron charge density (C/m^2) using Fermi-Dirac statistics.
    """
    temperature = 1.0  # Temperature in Kelvin (adjust as needed)
    kT = const.k * temperature  # Thermal energy

    density_2d = np.zeros((Nx, Ny))
    num_states = eigenvectors_2d.shape[2]

    for i in range(num_states):
        # Fermi-Dirac distribution
        fermi_dirac = 1 / (1 + np.exp((eigenvalues[i] - fermi_level) / kT))
        density_2d += 2 * fermi_dirac * np.abs(eigenvectors_2d[:, :, i]) ** 2

    charge_density_2d = -e * density_2d
    return charge_density_2d


# --- Poisson Solver (2D) ---
def solve_poisson_2d(charge_density_2d):
    """
    Solves the 2D Poisson equation: laplacian(phi) = -rho / epsilon
    Returns the 2D electrostatic potential phi (Volts).
    Uses finite differences and assumes Dirichlet boundary conditions (phi=0 on boundary).
    """
    rho_flat = charge_density_2d.flatten(order="C")

    # Construct the 2D Laplacian matrix (similar to Hamiltonian kinetic part)
    diag = (-2 / dx**2 - 2 / dy**2) * np.ones(N_total)
    offdiag_x = (1 / dx**2) * np.ones(N_total)
    offdiag_y = (1 / dy**2) * np.ones(N_total)

    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    # Adjust off-diagonals at boundaries
    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0  # Corrected index

    A = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    # Right-hand side vector b = -rho / epsilon
    b = -rho_flat / epsilon

    # Apply Dirichlet boundary conditions (phi=0 on all edges)
    # We can do this by modifying rows/columns corresponding to boundary points
    A = A.tolil()
    b_modified = b.copy()  # Modify a copy

    boundary_indices = []
    # Indices for x=0 and x=Nx-1 boundaries
    boundary_indices.extend(range(0, N_total, Ny))  # i=0, all j
    boundary_indices.extend(range(Ny - 1, N_total, Ny))  # i=Nx-1, all j
    # Indices for y=0 and y=Ny-1 boundaries
    boundary_indices.extend(range(1, Ny - 1))  # j=0, 0<i<Nx-1
    boundary_indices.extend(range(N_total - Ny + 1, N_total - 1))  # j=Ny-1, 0<i<Nx-1

    # Remove duplicates and sort
    boundary_indices = sorted(list(set(boundary_indices)))

    for idx in boundary_indices:
        A.rows[idx] = [idx]  # Set row to identity
        A.data[idx] = [1.0]
        b_modified[idx] = 0.0  # Set RHS to 0 for boundary condition

    A = A.tocsc()

    # Solve the linear system A * phi = b_modified
    try:
        phi_flat = spla.spsolve(A, b_modified)
    except spla.MatrixRankWarning:
        print(
            "Warning: Poisson matrix is singular or near-singular. Using iterative solver."
        )
        phi_flat, info = spla.gmres(A, b_modified, tol=1e-8, maxiter=2 * N_total)
        # Example iterative solver
        if info != 0:
            print(f"Poisson solver (GMRES) did not converge (info={info}).")
            phi_flat = np.zeros_like(rho_flat)  # Fallback
    except Exception as e:
        print(f"Poisson solver failed: {e}")
        phi_flat = np.zeros_like(rho_flat)  # Fallback

    phi_2d = phi_flat.reshape((Nx, Ny), order="C")
    return phi_2d  # Electrostatic potential in Volts


# --- Self-Consistent Iteration (2D) ---
def self_consistent_solver_2d(voltages, fermi_level, max_iter=50, tol=1e-5, mixing=0.1):
    """
    Performs the self-consistent 2D Schrödinger-Poisson calculation.
    """
    print("Starting 2D self-consistent calculation...")
    start_time = time.time()

    # Initial guess
    electrostatic_potential_V = np.zeros((Nx, Ny))  # Potential in Volts
    external_potential_J = get_external_potential(X, Y, voltages)  # Potential in Joules

    # Adaptive mixing parameters
    initial_mixing = mixing
    min_mixing = 0.01
    mixing_decay_rate = 0.9  # Reduce mixing by this factor if not converging
    previous_potential_diff_norm = float('inf')  # Initialize to a large value

    for i in range(max_iter):
        iter_start_time = time.time()
        print(f"Iteration {i + 1}/{max_iter}")

        # 1. Calculate total potential energy V = V_ext + (-e * phi)
        total_potential_J = external_potential_J - e * electrostatic_potential_V

        # 2. Solve Schrödinger equation
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(total_potential_J)
        if not eigenvalues.size:
            print("Error in Schrödinger solver. Aborting.")
            return None, None, None, None  # Indicate failure

        # 3. Calculate charge density
        new_charge_density = calculate_charge_density_2d(
            eigenvalues, eigenvectors_2d, fermi_level
        )

        # 4. Solve Poisson equation
        new_electrostatic_potential_V = solve_poisson_2d(new_charge_density)

        # 5. Check for convergence (using norm of potential difference)
        potential_diff_norm = np.linalg.norm(
            new_electrostatic_potential_V - electrostatic_potential_V
        ) * np.sqrt(dx * dy)
        print(f"  Potential difference norm: {potential_diff_norm:.3e}")
        if potential_diff_norm < tol:
            print(f"Converged after {i + 1} iterations.")
            electrostatic_potential_V = new_electrostatic_potential_V
            break

        # Check if potential difference is increasing (not converging)
        if i > 0 and potential_diff_norm > previous_potential_diff_norm:
            mixing *= mixing_decay_rate
            if mixing < min_mixing:
                mixing = min_mixing
            print(f"  Reducing mixing parameter to {mixing:.3f}")

        # 6. Mix potential for stability
        electrostatic_potential_V = electrostatic_potential_V + mixing * (
            new_electrostatic_potential_V - electrostatic_potential_V
        )

        iter_end_time = time.time()
        print(f"  Iteration time: {iter_end_time - iter_start_time:.2f} seconds")

        previous_potential_diff_norm = potential_diff_norm

    else:  # Loop finished without break
        print(f"Warning: Did not converge after {max_iter} iterations.")

    end_time = time.time()
    print(f"Total self-consistent loop time: {end_time - start_time:.2f} seconds")

    # Final calculation with converged potential
    final_total_potential_J = external_potential_J - e * electrostatic_potential_V
    final_eigenvalues, final_eigenvectors_2d = solve_schrodinger_2d(
        final_total_potential_J
    )
    final_charge_density = calculate_charge_density_2d(
        final_eigenvalues, final_eigenvectors_2d, fermi_level
    )

    return (
        final_total_potential_J,
        final_charge_density,
        final_eigenvalues,
        final_eigenvectors_2d,
    )


# --- Main Execution ---
if __name__ == "__main__":
    # Define applied voltages for the 2D double dot device (Volts)
    applied_voltages = {
        "P1": 0.0,  # Plunger 1 voltage (swept)
        "P2": 0.0,  # Plunger 2 voltage (swept)
        "B1": 0.15,  # Left barrier voltage
        "B2": 0.20,  # Center barrier voltage
        "B3": 0.15,  # Right barrier voltage
    }

    # Define Fermi level (relative to minimum external potential)
    initial_ext_pot_J = get_external_potential(X, Y, applied_voltages)
    fermi_level_J = (
        np.min(initial_ext_pot_J) + 0.05 * e
    )  # Adjusted Fermi level

    # --- Coulomb Diamond Sweep Parameters ---
    gate_1_name = "P1"
    gate_2_name = "P2"
    gate_1_voltages = np.linspace(-0.3, 0.1, 51)  # Example range
    gate_2_voltages = np.linspace(-0.3, 0.1, 51)  # Example range

    # --- Data Storage ---
    electron_numbers = np.zeros(
        (len(gate_1_voltages), len(gate_2_voltages))
    )  # 2D array to store electron numbers

    # --- Sweep Simulation ---
    start_time = time.time()
    for i, v1 in enumerate(gate_1_voltages):
        for j, v2 in enumerate(gate_2_voltages):
            print(
                f"Running simulation for {gate_1_name} = {v1:.3f} V, {gate_2_name} = {v2:.3f} V"
            )
            applied_voltages[gate_1_name] = v1
            applied_voltages[gate_2_name] = v2

            # Run the self-consistent solver
            total_potential, charge_density, eigenvalues, eigenvectors = (
                self_consistent_solver_2d(
                    applied_voltages,
                    fermi_level_J,
                    max_iter=30,
                    tol=1e-4,
                    mixing=0.05,
                )
            )

            if total_potential is not None:
                # Calculate the number of electrons in the dot
                electron_number = np.sum(charge_density * dx * dy) / (-e)
                electron_numbers[i, j] = electron_number
                print(f"  -> Number of electrons in the dot: {electron_number:.3f}")
            else:
                print(
                    f"  -> Simulation failed for {gate_1_name} = {v1:.3f} V, {gate_2_name} = {v2:.3f} V. Skipping point."
                )
                electron_numbers[i, j] = np.nan

    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")

    # --- Plotting Coulomb Diamonds ---
    plt.figure(figsize=(8, 6))
    extent = [
        gate_1_voltages[0],
        gate_1_voltages[-1],
        gate_2_voltages[0],
        gate_2_voltages[-1],
    ]  # Define the plot boundaries
    plt.imshow(
        electron_numbers.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )  # Use electron_numbers.T to match the axis orientation
    plt.colorbar(label="Number of Electrons")
    plt.xlabel(f"Gate Voltage {gate_1_name} (V)")
    plt.ylabel(f"Gate Voltage {gate_2_name} (V)")
    plt.title("Coulomb Diamonds")
    plt.tight_layout()

    plot_filename = "coulomb_diamonds.png"
    plt.savefig(plot_filename)
    print(f"Coulomb diamond plot saved to {plot_filename}")
```
