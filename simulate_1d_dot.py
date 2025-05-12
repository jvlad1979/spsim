#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
1D Schrödinger-Poisson simulator for a semiconductor quantum dot device.
"""

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# --- Physical Constants ---
hbar = const.hbar
m_e = const.m_e
e = const.e
epsilon_0 = const.epsilon_0

# --- Material Parameters (e.g., GaAs) ---
m_eff = 0.067 * m_e  # Effective mass
eps_r = 12.9         # Relative permittivity
epsilon = eps_r * epsilon_0

# --- Simulation Grid ---
L = 200e-9  # Length of the simulation domain (m)
N = 401     # Number of grid points (odd number for centered features)
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# --- Device Parameters ---
# Example: A simple potential well defined by gate voltages
# These would define regions where voltages are applied
# For now, let's define a simple external potential function
def get_external_potential(x, voltages):
    """
    Calculates the external potential profile for a double dot device
    based on gate voltages applied to plungers and barriers.
    Uses Gaussian profiles for each gate's influence.
    """
    potential = np.zeros_like(x)

    # Define gate parameters (centers and widths in meters)
    # Adjust these based on desired dot separation and size
    gate_std_dev = 15e-9 # Standard deviation for all gates
    p1_center = L * 0.35
    p2_center = L * 0.65
    b1_center = L * 0.20 # Left barrier
    b2_center = L * 0.50 # Center barrier
    b3_center = L * 0.80 # Right barrier

    # Helper function for Gaussian potential shape
    # Amplitude is voltage * e (energy), sign depends on gate type
    def gaussian_potential(center, std_dev, amplitude):
        return amplitude * np.exp(-(x - center)**2 / (2 * std_dev**2))

    # Add potential contributions from each gate based on applied voltages
    # Plunger voltages (typically negative) lower the potential energy
    potential += gaussian_potential(p1_center, gate_std_dev, voltages.get('P1', 0.0) * e)
    potential += gaussian_potential(p2_center, gate_std_dev, voltages.get('P2', 0.0) * e)

    # Barrier voltages (typically positive) raise the potential energy
    # Note: Here we assume positive voltage values map to positive potential energy barriers
    potential += gaussian_potential(b1_center, gate_std_dev, voltages.get('B1', 0.0) * e)
    potential += gaussian_potential(b2_center, gate_std_dev, voltages.get('B2', 0.0) * e)
    potential += gaussian_potential(b3_center, gate_std_dev, voltages.get('B3', 0.0) * e)

    # Optional: Add a constant background potential offset if needed
    # potential += background_offset * e

    return potential

# --- Schrödinger Solver ---
def solve_schrodinger(potential):
    """
    Solves the 1D time-independent Schrödinger equation for the given potential.
    Returns eigenvalues (energies) and eigenvectors (wavefunctions).
    """
    # Hamiltonian matrix: H = -hbar^2/(2*m_eff) * d^2/dx^2 + V(x)
    diag = hbar**2 / (m_eff * dx**2) + potential
    offdiag = -hbar**2 / (2 * m_eff * dx**2) * np.ones(N - 1)
    # Using sparse matrix for efficiency, especially for larger N
    H = sp.diags([offdiag, diag, offdiag], [-1, 0, 1], shape=(N, N), format='csc')

    # Find the lowest few eigenvalues and eigenvectors
    # k: number of eigenstates to find
    # sigma: find eigenvalues near this value (e.g., near the potential minimum)
    # which='LM': find eigenvalues with the largest magnitude (use sigma for specific range)
    # Using shift-invert mode (sigma) is generally better for finding lowest states
    try:
        # Find eigenvalues near the minimum of the potential
        num_eigenstates = 10 # Number of eigenstates to compute
        # Use sparse solver spla.eigsh for Hermitian matrices
        # Using which='SM' (Smallest Magnitude) without sigma can be more robust
        eigenvalues, eigenvectors = spla.eigsh(H, k=num_eigenstates, which='SM')
        # eigsh returns sorted eigenvalues
    except Exception as e:
        print(f"Eigenvalue solver failed: {e}")
        # Fallback or error handling needed
        # For now, return empty arrays
        return np.array([]), np.empty((N, 0))

    # Normalize eigenvectors
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:, i] /= np.sqrt(np.sum(np.abs(eigenvectors[:, i])**2) * dx)

    return eigenvalues, eigenvectors # Eigenvalues in J, eigenvectors are dimensionless

# --- Charge Density Calculation ---
def calculate_charge_density(eigenvalues, eigenvectors, fermi_level):
    """
    Calculates the 1D electron charge density.
    Assumes zero temperature (Fermi-Dirac distribution is a step function).
    """
    density = np.zeros(N)
    # Factor of 2 for spin degeneracy
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < fermi_level:
            density += 2 * np.abs(eigenvectors[:, i])**2
        else:
            break # Eigenvalues are sorted

    # Charge density (negative for electrons)
    charge_density = -e * density # Coulombs per meter (C/m)
    return charge_density

# --- Poisson Solver ---
def solve_poisson(charge_density):
    """
    Solves the 1D Poisson equation: d^2(phi)/dx^2 = -rho / epsilon
    Returns the electrostatic potential phi.
    Uses finite differences and assumes Dirichlet boundary conditions (phi=0 at ends).
    """
    # Discretized Poisson equation: (phi_{i+1} - 2*phi_i + phi_{i-1}) / dx^2 = -rho_i / epsilon
    # This forms a linear system A * phi = b
    # A is a tridiagonal matrix: [1, -2, 1] / dx^2
    diag = -2 / dx**2 * np.ones(N)
    offdiag = 1 / dx**2 * np.ones(N - 1)
    A = sp.diags([offdiag, diag, offdiag], [-1, 0, 1], format='csc')

    # Right-hand side vector b = -rho / epsilon
    b = -charge_density / epsilon

    # Apply boundary conditions (phi[0] = 0, phi[N-1] = 0)
    # Modify A and b to enforce boundary conditions
    # For Dirichlet phi=0:
    # Row 0: Set A[0,0]=1, A[0,1]=0, b[0]=0
    # Row N-1: Set A[N-1,N-1]=1, A[N-1,N-2]=0, b[N-1]=0
    # A more robust way using sparse matrices:
    A = A.tolil() # Convert to LIL format for easier modification
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[N - 1, N - 1] = 1.0
    A[N - 1, N - 2] = 0.0
    b[0] = 0.0
    b[N - 1] = 0.0
    A = A.tocsc() # Convert back to CSC format for efficient solving

    # Solve the linear system A * phi = b
    try:
        phi = spla.spsolve(A, b)
    except spla.MatrixRankWarning:
        print("Warning: Matrix is singular or near-singular. Using least squares.")
        phi, info = spla.lgmres(A, b) # Example iterative solver
        if info != 0:
            print(f"Poisson solver did not converge (info={info}).")
            # Fallback: Use pseudo-inverse or handle error
            phi = np.zeros_like(charge_density) # Return zero potential as fallback

    return phi # Electrostatic potential in Volts

# --- Self-Consistent Iteration ---
def self_consistent_solver(voltages, fermi_level, max_iter=100, tol=1e-6, mixing=0.1):
    """
    Performs the self-consistent Schrödinger-Poisson calculation.
    """
    print("Starting self-consistent calculation...")

    # Initial guess: no charge, potential is just the external potential
    charge_density = np.zeros(N)
    electrostatic_potential_V = np.zeros(N) # Potential in Volts

    external_potential_J = get_external_potential(x, voltages) # Potential in Joules

    for i in range(max_iter):
        print(f"Iteration {i+1}/{max_iter}")

        # 1. Calculate total potential energy V = V_ext + (-e * phi)
        total_potential_J = external_potential_J - e * electrostatic_potential_V

        # 2. Solve Schrödinger equation
        eigenvalues, eigenvectors = solve_schrodinger(total_potential_J)
        if not eigenvalues.size:
            print("Error in Schrödinger solver. Aborting.")
            return None, None, None, None # Indicate failure

        # 3. Calculate charge density
        new_charge_density = calculate_charge_density(eigenvalues, eigenvectors, fermi_level)

        # 4. Solve Poisson equation
        new_electrostatic_potential_V = solve_poisson(new_charge_density)

        # 5. Check for convergence
        # Convergence check based on the change in potential
        potential_diff = np.linalg.norm(new_electrostatic_potential_V - electrostatic_potential_V) * dx
        print(f"  Potential difference norm: {potential_diff:.3e}")
        if potential_diff < tol:
            print(f"Converged after {i+1} iterations.")
            electrostatic_potential_V = new_electrostatic_potential_V
            charge_density = new_charge_density
            break

        # 6. Mix new and old potential/density for stability
        # Simple linear mixing: potential = old_potential + mixing * (new_potential - old_potential)
        electrostatic_potential_V = electrostatic_potential_V + mixing * (new_electrostatic_potential_V - electrostatic_potential_V)
        # Alternatively, mix charge density:
        # charge_density = charge_density + mixing * (new_charge_density - charge_density)

        # Update charge density for the next Poisson solve if mixing potential
        # If mixing density, update potential based on mixed density
        charge_density = calculate_charge_density(eigenvalues, eigenvectors, fermi_level) # Recalculate based on current states
                                                                                        # This might need refinement depending on mixing strategy

    else: # Loop finished without break
        print(f"Warning: Did not converge after {max_iter} iterations.")

    # Final results
    final_total_potential_J = external_potential_J - e * electrostatic_potential_V
    final_eigenvalues, final_eigenvectors = solve_schrodinger(final_total_potential_J)
    final_charge_density = calculate_charge_density(final_eigenvalues, final_eigenvectors, fermi_level)

    return final_total_potential_J, final_charge_density, final_eigenvalues, final_eigenvectors


# --- Main Execution ---
if __name__ == "__main__":
    # Define applied voltages for the double dot device (Volts)
    # Adjust these values to shape the double well potential
    applied_voltages = {
        'P1': -0.25,  # Plunger 1 voltage (negative to create well)
        'P2': -0.25,  # Plunger 2 voltage
        'B1': 0.1,   # Left barrier voltage (positive to create barrier)
        'B2': 0.15,  # Center barrier voltage
        'B3': 0.1,   # Right barrier voltage
    }

    # Define Fermi level (chemical potential of the electron reservoir)
    # Example: Set slightly above the lowest potential energy
    ext_pot = get_external_potential(x, applied_voltages)
    fermi_level_J = np.min(ext_pot) + 0.05 * e # Example: 50 meV above min potential (in Joules)

    # Run the self-consistent solver
    total_potential, charge_density, eigenvalues, eigenvectors = self_consistent_solver(
        applied_voltages, fermi_level_J, max_iter=50, tol=1e-5, mixing=0.1
    )

    # --- Plotting Results ---
    if total_potential is not None:
        print("Plotting results...")
        plt.figure(figsize=(12, 8))

        # Plot 1: Potential Energy Profile
        plt.subplot(2, 1, 1)
        plt.plot(x * 1e9, total_potential / e, label='Total Potential (eV)')
        plt.plot(x * 1e9, get_external_potential(x, applied_voltages) / e, '--', label='External Potential (eV)')
        plt.axhline(fermi_level_J / e, color='r', linestyle=':', label=f'Fermi Level ({fermi_level_J/e:.3f} eV)')
        plt.xlabel("Position (nm)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Self-Consistent Potential Profile & Gate Layout")

        # --- Add Gate Visualization ---
        # Re-define gate parameters used in get_external_potential for visualization
        gate_std_dev_vis = 15e-9 # Standard deviation for all gates
        p1_center_vis = L * 0.35
        p2_center_vis = L * 0.65
        b1_center_vis = L * 0.20 # Left barrier
        b2_center_vis = L * 0.50 # Center barrier
        b3_center_vis = L * 0.80 # Right barrier

        # Draw vertical lines for gate centers
        plt.axvline(x=p1_center_vis * 1e9, color='blue', linestyle='--', alpha=0.7, label='P1 Center')
        plt.axvline(x=p2_center_vis * 1e9, color='cyan', linestyle='--', alpha=0.7, label='P2 Center')
        plt.axvline(x=b1_center_vis * 1e9, color='red', linestyle=':', alpha=0.7, label='B1 Center')
        plt.axvline(x=b2_center_vis * 1e9, color='magenta', linestyle=':', alpha=0.7, label='B2 Center')
        plt.axvline(x=b3_center_vis * 1e9, color='orange', linestyle=':', alpha=0.7, label='B3 Center')
        # --- End Gate Visualization ---

        plt.legend(loc='upper right') # Update legend location if needed
        plt.grid(True)

        # Plot 2: Charge Density and Wavefunctions
        plt.subplot(2, 1, 2)
        # Convert charge density C/m to electrons/nm (charge density / (-e) * 1e-9)
        charge_density_per_nm = charge_density / (-e) * 1e-9
        plt.plot(x * 1e9, charge_density_per_nm, label='Electron Density (electrons/nm)')

        # Plot lowest few wavefunctions (scaled for visibility)
        plot_lines = [plt.gca().lines[-1]] # Start with the density plot line
        if eigenvalues.size > 0:
            # Determine a suitable scale based on the density plot range or a fixed value
            density_min, density_max = np.min(charge_density_per_nm), np.max(charge_density_per_nm)
            density_range = max(density_max - density_min, 1e-9) # Avoid division by zero if density is flat
            scale = density_range * 2.0 # Scale factor for wavefunctions relative to density range

            for i in range(min(5, len(eigenvalues))): # Plot up to 5 lowest states
                # Shift wavefunction vertically by its energy (in eV)
                # Probability density |psi|^2 / nm, scaled for visibility
                psi_scaled = np.abs(eigenvectors[:, i])**2 * (dx / 1e-9)
                # Normalize the peak of the scaled psi^2 to the 'scale' value for consistent plotting
                psi_scaled *= scale / np.max(psi_scaled) if np.max(psi_scaled) > 1e-9 else 1.0
                psi_plot = (eigenvalues[i] / e) + psi_scaled # Shift baseline to energy level
                line, = plt.plot(x * 1e9, psi_plot, label=f'E{i} ({eigenvalues[i]/e:.3f} eV)')
                plot_lines.append(line)

        plt.xlabel("Position (nm)")
        plt.ylabel("Electron Density / Scaled |Ψ|^²")
        plt.title("Electron Density and Wavefunctions")
        plt.legend(loc='upper right')
        plt.grid(True)

        # Adjust ylim dynamically to fit all plotted data (density + wavefunctions)
        min_y_val = min(line.get_ydata().min() for line in plot_lines)
        max_y_val = max(line.get_ydata().max() for line in plot_lines)
        plt.ylim(min_y_val - 0.1 * abs(min_y_val), max_y_val + 0.1 * abs(max_y_val)) # Add 10% margin

        plt.tight_layout()
        plot_filename = "simulation_results.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    else:
        print("Self-consistent calculation failed. No results to plot.")
