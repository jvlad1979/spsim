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
    Calculates the external potential profile based on gate voltages.
    This is a placeholder and needs specific implementation based on device geometry.
    Example: A simple quantum well.
    """
    V0 = -0.1 * e # Depth of the well in Joules (e.g., 0.1 eV)
    well_width = 50e-9
    well_center = L / 2
    potential = np.zeros_like(x)
    potential[np.abs(x - well_center) < well_width / 2] = V0
    # Add contributions from voltages here based on actual gate layout
    # For example: potential += voltages['plunger1'] * gate_profile_plunger1(x)
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
    # Define applied voltages (example)
    # This needs to match the structure expected by get_external_potential
    applied_voltages = {
        'plunger1': 0.0, # Volts
        'barrier1': 0.0, # Volts
        # Add other gates as needed
    }

    # Define Fermi level (chemical potential of the electron reservoir)
    # Example: Set slightly above the lowest potential energy
    ext_pot = get_external_potential(x, applied_voltages)
    fermi_level_J = np.min(ext_pot) + 0.01 * e # Example: 10 meV above min potential (in Joules)

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
        plt.title("Self-Consistent Potential Profile")
        plt.legend()
        plt.grid(True)

        # Plot 2: Charge Density and Wavefunctions
        plt.subplot(2, 1, 2)
        # Convert charge density C/m to electrons/nm (charge density / (-e) * 1e-9)
        charge_density_per_nm = charge_density / (-e) * 1e-9
        plt.plot(x * 1e9, charge_density_per_nm, label='Electron Density (electrons/nm)')

        # Plot lowest few wavefunctions (scaled for visibility)
        if eigenvalues.size > 0:
            ymin, ymax = plt.ylim()
            scale = (ymax - ymin) * 0.1 # Scaling factor for wavefunctions
            for i in range(min(5, len(eigenvalues))): # Plot up to 5 lowest states
                # Shift wavefunction vertically by its energy (in eV)
                psi_plot = (eigenvalues[i] / e) + scale * np.abs(eigenvectors[:, i])**2 * (dx / 1e-9) # Probability density |psi|^2 / nm
                plt.plot(x * 1e9, psi_plot, label=f'E{i} ({eigenvalues[i]/e:.3f} eV)')

        plt.xlabel("Position (nm)")
        plt.ylabel("Electron Density (electrons/nm) / Wavefunctions")
        plt.title("Electron Density and Wavefunctions")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.ylim(ymin, ymax + scale * 2) # Adjust ylim to show wavefunctions

        plt.tight_layout()
        plt.show()
    else:
        print("Self-consistent calculation failed. No results to plot.")
