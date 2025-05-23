#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Schrödinger-Poisson simulator for a semiconductor quantum dot device.
Modified to simulate pinch-off characteristics by sweeping a gate voltage.
"""

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numpy.fft as fft  # Added for spectral methods

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Set the font family to serif
        "font.serif": ["Times New Roman"],  # Specify the serif font
        "font.size": 12,  # Set the default font size - Adjusted for better readability in subplots
    }
)

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

    # Optional: Add confinement in y-direction (e.g., parabolic)
    # potential += 0.5 * m_eff * (omega_y**2) * (Y - Ly/2)**2

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
        Uses finite differences and assumes Dirichlet boundary conditions (phi=0 on
    boundary).
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


# --- Self-Consistent Iteration (2D) ---
def self_consistent_solver_2d(
    voltages,
    fermi_level,
    max_iter=50,
    tol=1e-5,
    mixing=0.1,
    poisson_solver_type="finite_difference",
):
    """
    Performs the self-consistent 2D Schrödinger-Poisson calculation.
    Allows choosing the Poisson solver ('finite_difference' or 'spectral').
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
    previous_potential_diff_norm = float("inf")  # Initialize to a large value

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
        "P1": -0.20,  # Plunger 1 voltage
        "P2": -0.20,  # Plunger 2 voltage
        "B1": 0.15,  # Left barrier voltage
        "B2": 0.20,  # Center barrier voltage - THIS WILL BE SWEPT
        "B3": 0.15,  # Right barrier voltage
    }

    # Define Fermi level (relative to minimum external potential)
    # Calculate based on initial voltages, but keep constant during sweep
    initial_ext_pot_J = get_external_potential(X, Y, applied_voltages)
    fermi_level_J = np.min(initial_ext_pot_J) + 0.05 * e  # Adjusted Fermi level

    # --- Pinch-off Sweep Parameters ---
    gate_to_sweep = "B2"
    # Define voltage range for the sweep (e.g., from barrier fully open to closed)
    sweep_voltages = np.linspace(-0.3, 0.3, 101)  # Finer sweep range and more points
    barrier_potentials = []
    estimated_currents = []
    electron_numbers = []  # Store the number of electrons in the dot

    # Parameter for current estimation exponential decay (e.g., 5 meV)
    current_decay_energy_scale_J = 0.005 * e

    # Find the index corresponding to the center of the swept gate
    # Gate B2 center: (Lx * 0.50, Ly * 0.5)
    center_x_idx = np.argmin(np.abs(x - Lx * 0.50))
    center_y_idx = np.argmin(np.abs(y - Ly * 0.50))
    # Define a small region around the gate center in x to find the minimum potential
    barrier_region_x_indices = slice(
        max(0, center_x_idx - 5), min(Nx, center_x_idx + 6)
    )

    print(f"\n--- Starting Pinch-off Sweep for Gate {gate_to_sweep} ---")
    print(f"Voltage range: {sweep_voltages[0]:.3f} V to {sweep_voltages[-1]:.3f} V")
    print(f"Number of points: {len(sweep_voltages)}")
    print(f"Fermi Level: {fermi_level_J / e:.4f} eV")
    print("-" * 50)

    sweep_start_time = time.time()

    # --- Loop through sweep voltages ---
    for v_sweep in sweep_voltages:
        print(f"\nRunning simulation for {gate_to_sweep} = {v_sweep:.3f} V")
        applied_voltages[gate_to_sweep] = v_sweep

        # Run the self-consistent solver
        # Add poisson_solver_type='finite_difference' or 'spectral' here
        total_potential, charge_density, eigenvalues, eigenvectors = (
            self_consistent_solver_2d(
                applied_voltages,
                fermi_level_J,
                max_iter=30,
                tol=1e-4,
                mixing=0.05,  # Use same parameters as before, adjust if needed
                poisson_solver_type="finite_difference",  # Choose the solver here
            )
        )

        if total_potential is not None:
            # Estimate barrier height: Find min potential along the center line (y=center_y_idx)
            # within the barrier region defined by barrier_region_x_indices.
            potential_slice = total_potential[barrier_region_x_indices, center_y_idx]
            min_barrier_potential_J = np.min(potential_slice)
            barrier_potentials.append(min_barrier_potential_J)
            print(
                f"  -> Minimum barrier potential: {min_barrier_potential_J / e:.4f} eV"
            )

            # Calculate the number of electrons in the dot
            electron_number = np.sum(charge_density * dx * dy) / (-e)
            electron_numbers.append(electron_number)
            print(f"  -> Number of electrons in the dot: {electron_number:.3f}")

        else:
            print(
                f"  -> Simulation failed for {gate_to_sweep} = {v_sweep:.3f} V. Skipping point."
            )
            barrier_potentials.append(np.nan)
            electron_numbers.append(np.nan)

    sweep_end_time = time.time()
    print("-" * 50)
    print(
        f"Pinch-off sweep finished in {sweep_end_time - sweep_start_time:.2f} seconds."
    )

    # --- Plotting Pinch-off Curve ---
    # Convert barrier potentials to eV for plotting
    barrier_potentials_eV = np.array(barrier_potentials) / e

    plt.figure(figsize=(8, 6))
    plt.plot(sweep_voltages, barrier_potentials_eV, marker="o", linestyle="-")
    plt.axhline(
        fermi_level_J / e,
        color="r",
        linestyle="--",
        label=f"Fermi Level ({fermi_level_J / e:.3f} eV)",
    )
    plt.xlabel(f"Gate Voltage {gate_to_sweep} (V)")
    plt.ylabel("Minimum Potential under Gate (eV)")
    plt.title(f"Pinch-off Curve: Barrier Potential vs. Gate Voltage ({gate_to_sweep})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f"pinchoff_curve_{gate_to_sweep}.png"
    plt.savefig(plot_filename)
    print(f"Pinch-off plot saved to {plot_filename}")

    # --- Plotting Number of Electrons vs Gate Voltage ---
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_voltages, electron_numbers, marker="o", linestyle="-")
    plt.xlabel(f"Gate Voltage {gate_to_sweep} (V)")
    plt.ylabel("Number of Electrons in Dot")
    plt.title(f"Number of Electrons vs. Gate Voltage ({gate_to_sweep})")
    plt.grid(True)
    plt.tight_layout()

    plot_filename_electrons = f"number_of_electrons_{gate_to_sweep}.png"
    plt.savefig(plot_filename_electrons)
    print(f"Number of electrons plot saved to {plot_filename_electrons}")

    # Optional: Plot the final potential landscape for the last voltage point
    if total_potential is not None:
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        contour = plt.contourf(
            X * 1e9, Y * 1e9, total_potential / e, levels=50, cmap="viridis"
        )
        plt.colorbar(contour, label="Total Potential (eV)")
        # Add gate visualization (copied from simulate_2d_dot.py's plotting section)
        gate_std_dev_x_vis = 20e-9
        gate_std_dev_y_vis = 20e-9
        p1_center_vis = (Lx * 0.35, Ly * 0.5)
        p2_center_vis = (Lx * 0.65, Ly * 0.5)
        b1_center_vis = (Lx * 0.15, Ly * 0.5)
        b2_center_vis = (Lx * 0.50, Ly * 0.5)  # Center of swept gate
        b3_center_vis = (Lx * 0.85, Ly * 0.5)
        gate_centers_vis = {
            "P1": p1_center_vis,
            "P2": p2_center_vis,
            "B1": b1_center_vis,
            "B2": b2_center_vis,
            "B3": b3_center_vis,
        }
        gate_colors = {
            "P1": "blue",
            "P2": "cyan",
            "B1": "red",
            "B2": "magenta",
            "B3": "orange",
        }
        gate_styles = {"P1": "--", "P2": "--", "B1": ":", "B2": ":", "B3": ":"}
        for name, center in gate_centers_vis.items():
            ellipse = patches.Ellipse(
                xy=(center[0] * 1e9, center[1] * 1e9),
                width=2 * gate_std_dev_x_vis * 1e9,
                height=2 * gate_std_dev_y_vis * 1e9,
                edgecolor=gate_colors[name],
                facecolor="none",
                linestyle=gate_styles[name],
                linewidth=1.5,
                label=name,
            )
            ax.add_patch(ellipse)
        handles, labels = ax.get_legend_handles_labels()
        patch_handles = [h for h in handles if isinstance(h, patches.Patch)]
        patch_labels = [
            l for h, l in zip(handles, labels) if isinstance(h, patches.Patch)
        ]
        if patch_handles:
            ax.legend(
                patch_handles,
                patch_labels,
                title="Gates",
                loc="upper right",
                fontsize="small",
            )

        # Highlight the line where barrier potential was measured
        plt.plot(
            [
                x[barrier_region_x_indices].min() * 1e9,
                x[barrier_region_x_indices].max() * 1e9,
            ],
            [y[center_y_idx] * 1e9, y[center_y_idx] * 1e9],
            color="yellow",
            linestyle="-",
            linewidth=2,
            label="Barrier Slice",
        )
        ax.legend()  # Update legend to include barrier slice line

        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.title(
            f"Final Potential Landscape ({gate_to_sweep} = {sweep_voltages[-1]:.3f} V)"
        )
        plt.axis("equal")
        plt.tight_layout()
        plot_filename_final_pot = f"final_potential_{gate_to_sweep}_sweep.png"
        plt.savefig(plot_filename_final_pot)
        print(f"Final potential plot saved to {plot_filename_final_pot}")

    else:
        print(
            "Self-consistent calculation failed for the last point. No final potential plot."
        )
