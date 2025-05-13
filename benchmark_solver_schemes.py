#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for testing various 2D Schrödinger-Poisson solver schemes.
This script evaluates the performance of different Poisson solvers (Finite Difference, Spectral)
and the impact of warm starts on convergence time.
"""

import numpy as np
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import numpy.fft as fft
# import cProfile # For more detailed profiling

# --- Physical Constants ---
hbar = const.hbar
m_e = const.m_e
e = const.e
epsilon_0 = const.epsilon_0

# --- Material Parameters (e.g., GaAs) ---
m_eff = 0.067 * m_e  # Effective mass
eps_r = 12.9  # Relative permittivity
epsilon = eps_r * epsilon_0

# --- Simulation Grid (Reduced size for faster benchmarking) ---
Lx = 150e-9  # Length of the simulation domain in x (m)
Ly = 100e-9  # Length of the simulation domain in y (m)
Nx = 45  # Number of grid points in x (reduced from 75 for speed)
Ny = 30  # Number of grid points in y (reduced from 50 for speed)
N_total = Nx * Ny

x_coords = np.linspace(0, Lx, Nx)
y_coords = np.linspace(0, Ly, Ny)
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]
X, Y = np.meshgrid(
    x_coords, y_coords, indexing="ij"
)  # 'ij' indexing matches matrix layout

# --- Spectral Method Setup ---
kx = 2 * np.pi * fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * fft.fftfreq(Ny, d=dy)
Kx_grid, Ky_grid = np.meshgrid(kx, ky, indexing='ij')
K_sq = Kx_grid**2 + Ky_grid**2
# K_sq[0, 0] is handled in the spectral Poisson solver to avoid division by zero.

# --- Device Parameters ---
def get_external_potential(X_grid, Y_grid, voltages):
    """
    Calculates the 2D external potential profile based on gate voltages.
    Uses 2D Gaussian profiles for gate influence. (Copied from simulate_2d_dot.py)
    """
    potential = np.zeros_like(X_grid)  # Potential in Joules

    gate_std_dev_x = 20e-9
    gate_std_dev_y = 20e-9

    p1_center = (Lx * 0.35, Ly * 0.5)
    p2_center = (Lx * 0.65, Ly * 0.5)
    b1_center = (Lx * 0.15, Ly * 0.5)
    b2_center = (Lx * 0.50, Ly * 0.5)
    b3_center = (Lx * 0.85, Ly * 0.5)

    def gaussian_potential_2d(center_x, center_y, std_dev_x, std_dev_y, amplitude):
        return amplitude * np.exp(
            -(
                (X_grid - center_x) ** 2 / (2 * std_dev_x**2)
                + (Y_grid - center_y) ** 2 / (2 * std_dev_y**2)
            )
        )

    potential += gaussian_potential_2d(
        p1_center[0], p1_center[1], gate_std_dev_x, gate_std_dev_y, voltages.get("P1", 0.0) * e
    )
    potential += gaussian_potential_2d(
        p2_center[0], p2_center[1], gate_std_dev_x, gate_std_dev_y, voltages.get("P2", 0.0) * e
    )
    potential += gaussian_potential_2d(
        b1_center[0], b1_center[1], gate_std_dev_x, gate_std_dev_y, voltages.get("B1", 0.0) * e
    )
    potential += gaussian_potential_2d(
        b2_center[0], b2_center[1], gate_std_dev_x, gate_std_dev_y, voltages.get("B2", 0.0) * e
    )
    potential += gaussian_potential_2d(
        b3_center[0], b3_center[1], gate_std_dev_x, gate_std_dev_y, voltages.get("B3", 0.0) * e
    )
    return potential


# --- Schrödinger Solver (2D) ---
def solve_schrodinger_2d(potential_2d):
    """
    Solves the 2D time-independent Schrödinger equation.
    (Copied from simulate_2d_dot.py)
    Further exploration: Test different spla.eigsh parameters (e.g., tol, maxiter if applicable).
    """
    potential_flat = potential_2d.flatten(order="C")
    diag_terms = hbar**2 / (m_eff * dx**2) + hbar**2 / (m_eff * dy**2) + potential_flat
    offdiag_x_terms = -(hbar**2) / (2 * m_eff * dx**2) * np.ones(N_total)
    offdiag_y_terms = -(hbar**2) / (2 * m_eff * dy**2) * np.ones(N_total)

    diagonals = [diag_terms, offdiag_x_terms[:-1], offdiag_x_terms[:-1], offdiag_y_terms[:-Nx], offdiag_y_terms[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx): # Boundary condition adjustments for finite difference
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0

    H = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    try:
        num_eigenstates = 10
        eigenvalues, eigenvectors_flat = spla.eigsh(H, k=num_eigenstates, which="SM")
    except Exception as exc:
        print(f"Eigenvalue solver failed: {exc}")
        return np.array([]), np.empty((Nx, Ny, 0))

    eigenvectors = np.zeros((Nx, Ny, num_eigenstates))
    for i in range(num_eigenstates):
        psi_flat = eigenvectors_flat[:, i]
        norm = np.sqrt(np.sum(np.abs(psi_flat) ** 2) * dx * dy)
        eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")
    return eigenvalues, eigenvectors


# --- Charge Density Calculation (2D) ---
def calculate_charge_density_2d(eigenvalues, eigenvectors_2d, fermi_level):
    """
    Calculates the 2D electron charge density (C/m^2) at T=0K.
    (Copied from simulate_2d_dot.py)
    """
    density_2d = np.zeros((Nx, Ny))  # electrons/m^2
    num_states = eigenvectors_2d.shape[2]
    for i in range(num_states):
        if eigenvalues[i] < fermi_level:
            density_2d += 2 * np.abs(eigenvectors_2d[:, :, i]) ** 2  # Factor 2 for spin
        else:
            break
    charge_density_2d = -e * density_2d
    return charge_density_2d


# --- Poisson Solvers (2D) ---
def solve_poisson_2d_fd(charge_density_2d):
    """
    Solves 2D Poisson equation using Finite Differences with Dirichlet BCs (phi=0).
    (Copied from simulate_2d_dot.py)
    """
    rho_flat = charge_density_2d.flatten(order="C")
    diag_terms = (-2 / dx**2 - 2 / dy**2) * np.ones(N_total)
    offdiag_x_terms = (1 / dx**2) * np.ones(N_total)
    offdiag_y_terms = (1 / dy**2) * np.ones(N_total)

    diagonals = [diag_terms, offdiag_x_terms[:-1], offdiag_x_terms[:-1], offdiag_y_terms[:-Nx], offdiag_y_terms[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx): # Boundary condition adjustments
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
        phi_flat, info = spla.gmres(A, b_modified, tol=1e-8, maxiter=2 * N_total)
        if info != 0: phi_flat = np.zeros_like(rho_flat)
    except Exception:
        phi_flat = np.zeros_like(rho_flat)
    return phi_flat.reshape((Nx, Ny), order="C")


def solve_poisson_2d_spectral(charge_density_2d):
    """
    Solves 2D Poisson equation using Spectral Method (FFT) with periodic BCs.
    (Copied from simulate_2d_dot.py)
    """
    rho_k = fft.fft2(charge_density_2d)
    phi_k = np.zeros_like(rho_k, dtype=complex)
    
    # Avoid division by zero for K_sq[0,0] (DC component)
    # For periodic BCs, average potential is arbitrary; set phi_k[0,0]=0.
    K_sq_solver = K_sq.copy()
    if K_sq_solver[0, 0] == 0: # Should be true for standard fftfreq setup
        K_sq_solver[0, 0] = 1.0 # Avoid division by zero, phi_k[0,0] is set to 0 anyway

    phi_k = -rho_k / (epsilon * K_sq_solver)
    phi_k[0, 0] = 0.0  # Set DC component of potential to zero

    phi_2d = fft.ifft2(phi_k).real
    return phi_2d


def solve_poisson_2d_fem_stub(charge_density_2d):
    """Placeholder for a Finite Element Method Poisson solver."""
    print("FEM Poisson solver is not implemented.")
    # To implement FEM, one might use libraries like FEniCS or scikit-fem,
    # or write a custom simple FEM solver. This typically involves:
    # 1. Defining a mesh.
    # 2. Choosing basis functions (e.g., linear Lagrange elements).
    # 3. Assembling stiffness matrix and load vector from the weak form of Poisson's eq.
    #    Integral(grad(phi) . grad(v) dx) = Integral( (rho/epsilon) * v dx) for all test funcs v.
    # 4. Applying boundary conditions.
    # 5. Solving the resulting linear system.
    raise NotImplementedError("FEM Poisson solver needs implementation.")
    # return np.zeros_like(charge_density_2d)


# --- Benchmarking Self-Consistent Solver ---
def benchmark_sc_iteration(voltages, fermi_level, poisson_solver_func,
                           initial_potential_V=None, max_iter=30, tol=1e-4, mixing=0.1):
    """
    Performs a self-consistent Schrödinger-Poisson calculation and benchmarks components.
    Returns a dictionary of timings and the converged electrostatic potential.
    """
    timings = {
        "iterations": 0,
        "total_time": 0,
        "ext_potential_time": 0,
        "schrodinger_time_avg": 0,
        "charge_calc_time_avg": 0,
        "poisson_time_avg": 0,
        "schrodinger_times_per_iter": [],
        "charge_calc_times_per_iter": [],
        "poisson_times_per_iter": [],
        "converged": False
    }

    loop_start_time = time.time()

    # Initial guess for electrostatic potential (Volts)
    if initial_potential_V is None:
        electrostatic_potential_V = np.zeros((Nx, Ny))
    else:
        electrostatic_potential_V = initial_potential_V.copy()

    # Calculate external potential (once)
    ext_pot_start_time = time.time()
    external_potential_J = get_external_potential(X, Y, voltages)
    timings["ext_potential_time"] = time.time() - ext_pot_start_time

    for i in range(max_iter):
        timings["iterations"] = i + 1
        
        total_potential_J = external_potential_J - e * electrostatic_potential_V

        # 1. Solve Schrödinger equation
        sch_start_time = time.time()
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(total_potential_J)
        timings["schrodinger_times_per_iter"].append(time.time() - sch_start_time)
        if not eigenvalues.size:
            print("Error in Schrödinger solver during SC iteration. Aborting.")
            timings["total_time"] = time.time() - loop_start_time
            return timings, None # Indicate failure

        # 2. Calculate charge density
        cc_start_time = time.time()
        new_charge_density = calculate_charge_density_2d(eigenvalues, eigenvectors_2d, fermi_level)
        timings["charge_calc_times_per_iter"].append(time.time() - cc_start_time)

        # 3. Solve Poisson equation
        ps_start_time = time.time()
        new_electrostatic_potential_V = poisson_solver_func(new_charge_density)
        timings["poisson_times_per_iter"].append(time.time() - ps_start_time)

        # 4. Check for convergence
        potential_diff_norm = np.linalg.norm(new_electrostatic_potential_V - electrostatic_potential_V) * np.sqrt(dx * dy)
        if potential_diff_norm < tol:
            timings["converged"] = True
            electrostatic_potential_V = new_electrostatic_potential_V # Final update
            break
        
        # 5. Mix potential for stability
        electrostatic_potential_V += mixing * (new_electrostatic_potential_V - electrostatic_potential_V)
    else: # Loop finished without break (no convergence)
        print(f"Warning: Did not converge after {max_iter} iterations.")

    timings["total_time"] = time.time() - loop_start_time
    if timings["schrodinger_times_per_iter"]:
        timings["schrodinger_time_avg"] = np.mean(timings["schrodinger_times_per_iter"])
    if timings["charge_calc_times_per_iter"]:
        timings["charge_calc_time_avg"] = np.mean(timings["charge_calc_times_per_iter"])
    if timings["poisson_times_per_iter"]:
        timings["poisson_time_avg"] = np.mean(timings["poisson_times_per_iter"])
        
    return timings, electrostatic_potential_V


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Schrödinger-Poisson Solver Benchmark Script")
    print(f"Grid: Nx={Nx}, Ny={Ny}\n")

    applied_voltages_base = {
        "P1": -0.15, "P2": -0.15, "B1": 0.10, "B2": 0.15, "B3": 0.10,
    }
    # For perturbed test
    applied_voltages_perturbed = applied_voltages_base.copy()
    applied_voltages_perturbed["P1"] -= 0.01 # Small perturbation

    # Define Fermi level (relative to min external potential of base configuration)
    _ext_pot_J_base = get_external_potential(X, Y, applied_voltages_base)
    fermi_level_J = np.min(_ext_pot_J_base) + 0.02 * e  # 20 meV above min potential

    poisson_solver_methods = {
        "Finite Difference": solve_poisson_2d_fd,
        "Spectral (FFT)": solve_poisson_2d_spectral,
        # "FEM (Stub)": solve_poisson_2d_fem_stub, # Uncomment to test stub
    }

    benchmark_results = {}

    for name, solver_func in poisson_solver_methods.items():
        print(f"\n--- Benchmarking Poisson Solver: {name} ---")
        benchmark_results[name] = {}

        # 1. Cold Start
        print("Running: Cold Start (Base Voltages)")
        try:
            timings_cold, potential_cold = benchmark_sc_iteration(
                applied_voltages_base, fermi_level_J, solver_func
            )
            benchmark_results[name]["cold_start"] = timings_cold
            if not timings_cold["converged"]: potential_cold = None # Ensure no warm start if cold failed
        except NotImplementedError as e:
            print(f"Skipping {name} due to: {e}")
            benchmark_results[name]["cold_start"] = {"error": str(e)}
            potential_cold = None # Cannot proceed with this solver
            continue # Skip to next solver
        except Exception as e:
            print(f"Error during cold start for {name}: {e}")
            benchmark_results[name]["cold_start"] = {"error": str(e)}
            potential_cold = None
            continue


        # 2. Warm Start (Same Voltages)
        if potential_cold is not None:
            print("Running: Warm Start (Base Voltages, using converged potential from cold start)")
            timings_warm_same_V, _ = benchmark_sc_iteration(
                applied_voltages_base, fermi_level_J, solver_func,
                initial_potential_V=potential_cold
            )
            benchmark_results[name]["warm_start_same_V"] = timings_warm_same_V
        else:
            print("Skipping: Warm Start (Same Voltages) due to failed cold start.")
            benchmark_results[name]["warm_start_same_V"] = {"skipped": "Cold start failed"}

        # 3. Warm Start (Perturbed Voltages)
        if potential_cold is not None: # Use potential from original base voltage cold start
            print("Running: Warm Start (Perturbed Voltages, using converged potential from base cold start)")
            timings_warm_pert_V, _ = benchmark_sc_iteration(
                applied_voltages_perturbed, fermi_level_J, solver_func,
                initial_potential_V=potential_cold
            )
            benchmark_results[name]["warm_start_perturbed_V"] = timings_warm_pert_V
        else:
            print("Skipping: Warm Start (Perturbed Voltages) due to failed cold start.")
            benchmark_results[name]["warm_start_perturbed_V"] = {"skipped": "Cold start failed"}

    # --- Print Summary of Results ---
    print("\n\n--- Benchmark Summary ---")
    for method_name, results in benchmark_results.items():
        print(f"\nPoisson Method: {method_name}")
        for test_type, timings in results.items():
            if "error" in timings or "skipped" in timings:
                status = timings.get("error", timings.get("skipped", "Unknown issue"))
                print(f"  {test_type.replace('_', ' ').title()}: {status}")
                continue

            print(f"  {test_type.replace('_', ' ').title()}:")
            if "converged" in timings:
                 print(f"    Converged: {timings['converged']} in {timings['iterations']} iterations")
            print(f"    Total Time: {timings['total_time']:.3f} s")
            if timings.get('ext_potential_time', 0) > 0: # Only for first call in a series
                 print(f"    Ext. Potential Calc Time: {timings['ext_potential_time']:.3f} s")
            print(f"    Avg Schrödinger Time: {timings['schrodinger_time_avg']:.3f} s/iter")
            print(f"    Avg Charge Calc Time: {timings['charge_calc_time_avg']:.3f} s/iter")
            print(f"    Avg Poisson Time: {timings['poisson_time_avg']:.3f} s/iter")

    print("\n--- Notes ---")
    print("1. 'Avg Time' is per self-consistent iteration.")
    print("2. Spectral method uses periodic BCs; Finite Difference uses Dirichlet phi=0.")
    print("3. For more detailed profiling of specific functions, consider using cProfile:")
    print("   Example: python -m cProfile -s cumtime benchmark_solver_schemes.py")
    print("4. To test different Schrödinger solvers: modify 'solve_schrodinger_2d' or its parameters.")
    print("5. To implement FEM: replace 'solve_poisson_2d_fem_stub' with a working FEM solver.")

    print("\nBenchmark script finished.")
