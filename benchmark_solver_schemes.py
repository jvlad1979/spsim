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
import random
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
def solve_schrodinger_2d(potential_2d, solver_config):
    """
    Solves the 2D time-independent Schrödinger equation using a specified solver configuration.
    (Copied from simulate_2d_dot.py and modified)
    solver_config (dict): Configuration for the solver.
        Example: {'method': 'eigsh', 'params': {'k': 10, 'which': 'SM'}}
                 {'method': 'eigsh', 'params': {'k': 10, 'which': 'LM', 'use_sigma_min_potential': True}}
                 {'method': 'lobpcg', 'params': {'k': 10, 'use_random_X': True, 'tol': 1e-5, 'maxiter': 200}}
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

    method = solver_config.get('method', 'eigsh')
    params = solver_config.get('params', {})
    num_eigenstates = params.get('k', 10)

    eigenvalues = np.array([])
    eigenvectors_flat = np.empty((N_total, 0))

    try:
        if method == 'eigsh':
            eigsh_params = params.copy()
            if eigsh_params.pop('use_sigma_min_potential', False):
                sigma = np.min(potential_flat)
                # Ensure sigma is not exactly an eigenvalue, or too close to cause issues (optional refinement)
                # sigma -= 1e-9 * e # Small shift if needed
                eigsh_params['sigma'] = sigma
                if 'which' not in eigsh_params: # sigma requires which='LM' or 'LA' typically
                    eigsh_params['which'] = 'LM'
            
            # Remove non-eigsh specific params if any were added for other types
            eigsh_params.pop('use_random_X', None) 

            eigenvalues, eigenvectors_flat = spla.eigsh(H, **eigsh_params)

        elif method == 'lobpcg':
            lobpcg_params = params.copy()
            k_lobpcg = lobpcg_params.pop('k') # k is handled by X shape
            
            if lobpcg_params.pop('use_random_X', False):
                X_init = np.random.rand(N_total, k_lobpcg)
            else:
                # Default to random if no other X strategy is defined for LOBPCG
                X_init = np.random.rand(N_total, k_lobpcg)
            
            # LOBPCG finds largest eigenvalues by default, so use -H for smallest
            # Or, if a version of LOBPCG supports smallest directly, use that.
            # scipy.sparse.linalg.lobpcg has `largest=True` by default. We need smallest.
            # A common trick is to solve for -H, or use shift-invert if possible.
            # For simplicity, let's assume we want smallest magnitude, which LOBPCG can do with B=None, M=None.
            # However, LOBPCG is typically for generalized eigenvalue problems or when a good preconditioner M is available.
            # For standard Hx=ex, eigsh is often more direct for smallest.
            # Let's try with `largest=False`.
            
            # Ensure 'tol' and 'maxiter' are present with defaults if not in params
            if 'tol' not in lobpcg_params: lobpcg_params['tol'] = 1e-5
            if 'maxiter' not in lobpcg_params: lobpcg_params['maxiter'] = N_total // 2 # Heuristic

            eigenvalues, eigenvectors_flat = spla.lobpcg(H, X_init, largest=False, **lobpcg_params)
            # Sort eigenvalues and eigenvectors as LOBPCG might not return them sorted like eigsh
            idx_sort = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors_flat = eigenvectors_flat[:, idx_sort]


        else:
            raise ValueError(f"Unsupported Schrödinger solver method: {method}")

    except Exception as exc:
        print(f"Eigenvalue solver {method} failed: {exc}")
        return np.array([]), np.empty((Nx, Ny, 0))

    eigenvectors = np.zeros((Nx, Ny, num_eigenstates))
    # Ensure we don't try to access more eigenvectors than computed, esp. if solver returned fewer than k
    actual_computed_states = eigenvectors_flat.shape[1]
    for i in range(min(num_eigenstates, actual_computed_states)):
        psi_flat = eigenvectors_flat[:, i]
        # Normalization (eigsh usually returns normalized, LOBPCG should too with B=None, M=None)
        # Re-normalize for safety / consistency.
        norm_sq = np.sum(np.abs(psi_flat) ** 2) * dx * dy
        if norm_sq > 1e-12: # Avoid division by zero for zero vectors
            norm = np.sqrt(norm_sq)
            eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")
        else:
            eigenvectors[:, :, i] = psi_flat.reshape((Nx, Ny), order="C") # Keep as is if norm is zero
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
def benchmark_sc_iteration(voltages, fermi_level, poisson_solver_func, schrodinger_solver_config,
                           initial_potential_V=None, max_iter=30, tol=1e-4, mixing=0.1):
    """
    Performs a self-consistent Schrödinger-Poisson calculation and benchmarks components.
    `schrodinger_solver_config` is passed to `solve_schrodinger_2d`.
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
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(total_potential_J, schrodinger_solver_config)
        timings["schrodinger_times_per_iter"].append(time.time() - sch_start_time)
        if not eigenvalues.size or eigenvectors_2d.shape[2] == 0: # Check if any eigenvectors were returned
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

    N_SAMPLES = 100 # Number of random voltage samples
    NUM_EIGENSTATES = 10 # Default number of eigenstates to solve for

    # Define voltage ranges for random generation, aligned with charge stability simulations
    # Gate names: "P1", "P2", "B1", "B2", "B3"
    # P1, P2 swept in (-0.1, 0.0) in stability. B1=0.15, B2=0.20, B3=0.15.
    voltage_config = {
        "P1": {"range": (-0.1, 0.0), "perturb_std": 0.02}, # Typical plunger sweep range
        "P2": {"range": (-0.1, 0.0), "perturb_std": 0.02}, # Typical plunger sweep range
        "B1": {"range": (0.1, 0.2), "perturb_std": 0.02},  # Around fixed 0.15V
        "B2": {"range": (0.15, 0.25), "perturb_std": 0.02},# Around fixed 0.20V
        "B3": {"range": (0.1, 0.2), "perturb_std": 0.02},  # Around fixed 0.15V
    }

    def generate_random_voltages():
        voltages = {}
        for gate, config in voltage_config.items():
            voltages[gate] = random.uniform(config["range"][0], config["range"][1])
        return voltages

    def generate_perturbed_voltages(base_voltages):
        perturbed_voltages = {}
        for gate, base_val in base_voltages.items():
            perturb = random.gauss(0, voltage_config[gate]["perturb_std"])
            perturbed_val = base_val + perturb
            # Clamp to original range to avoid extreme values after perturbation
            min_v, max_v = voltage_config[gate]["range"]
            perturbed_voltages[gate] = np.clip(perturbed_val, min_v, max_v)
        return perturbed_voltages

    # Define Fermi level: For simplicity, calculate once based on 'average' expected potential
    # This could be refined to be sample-specific if needed.
    avg_voltages = {gate: np.mean(cfg["range"]) for gate, cfg in voltage_config.items()}
    _ext_pot_avg = get_external_potential(X, Y, avg_voltages)
    fermi_level_J = np.min(_ext_pot_avg) + 0.02 * e  # 20 meV above min of average potential

    schrodinger_solver_configs_to_test = [
        {"name": "eigsh_SM", "method": "eigsh", "params": {"k": NUM_EIGENSTATES, "which": "SM"}},
        {"name": "eigsh_LM_sigma_min_pot", "method": "eigsh",
         "params": {"k": NUM_EIGENSTATES, "which": "LM", "use_sigma_min_potential": True, "tol":1e-8}}, # Added tol for sigma mode
        # LOBPCG can be sensitive; ensure params are well-chosen or add later.
        # {"name": "lobpcg_randX", "method": "lobpcg",
        #  "params": {"k": NUM_EIGENSTATES, "use_random_X": True, "tol": 1e-7, "maxiter": (Nx*Ny)//4}},
    ]

    poisson_solver_methods = {
        "Finite Difference": solve_poisson_2d_fd,
        "Spectral (FFT)": solve_poisson_2d_spectral,
    }

    benchmark_summary_stats = {}

    def calculate_stats(timings_list):
        if not timings_list:
            # Ensure all expected keys are present even for empty lists
            stat_keys = ["iterations", "total_time", "ext_potential_time", 
                         "schrodinger_time_avg", "charge_calc_time_avg", "poisson_time_avg"]
            return {key: {"mean": np.nan, "std": np.nan, "count": 0, "converged_count":0} for key in stat_keys}

        stats = {}
        # converged_count is based on the number of items in timings_list, assuming only converged runs are added
        converged_runs = len(timings_list)

        for key in timings_list[0].keys(): # Use keys from the first valid timing dict
            if key.endswith("_per_iter") or key == "converged": # Skip these raw lists or boolean
                continue
            
            values = [d[key] for d in timings_list if isinstance(d.get(key), (int, float))]
            if values:
                stats[key] = {"mean": np.mean(values), "std": np.std(values), "count": len(values), "converged_count": converged_runs}
            else: # Should not happen if timings_list is not empty and contains valid numbers
                stats[key] = {"mean": np.nan, "std": np.nan, "count": 0, "converged_count": converged_runs}
        return stats

    for sch_config in schrodinger_solver_configs_to_test:
        sch_name = sch_config['name']
        benchmark_summary_stats[sch_name] = {}
        for poisson_name, poisson_solver_func in poisson_solver_methods.items():
            print(f"\n--- Benchmarking Schrödinger: {sch_name} | Poisson: {poisson_name} ---")
            current_config_stats = {}

            # --- Scenario 1: Random Voltages (Cold Starts) ---
            print(f"  Running Scenario: {N_SAMPLES} Random Cold Starts")
            cold_start_run_timings = []
            for i_sample in range(N_SAMPLES):
                random_voltages = generate_random_voltages()
                timings, _ = benchmark_sc_iteration(
                    random_voltages, fermi_level_J, poisson_solver_func,
                    schrodinger_solver_config=sch_config,
                    initial_potential_V=None
                )
                if timings["converged"]:
                    cold_start_run_timings.append(timings)
                # else:
                #     print(f"    Sample {i_sample+1} (Cold) for {sch_name}/{poisson_name} did not converge.")
            current_config_stats["random_cold_starts"] = calculate_stats(cold_start_run_timings)
            num_converged_cold = len(cold_start_run_timings)
            print(f"    Converged {num_converged_cold}/{N_SAMPLES} times for Random Cold Starts.")


            # --- Scenario 2: Perturbed Warm Starts ---
            print(f"  Running Scenario: {N_SAMPLES} Perturbed Warm Starts")
            perturbed_warm_start_run_timings = []
            converged_perturbed_warm = 0
            for i_sample in range(N_SAMPLES):
                base_random_voltages = generate_random_voltages()
                base_timings, potential_base = benchmark_sc_iteration(
                    base_random_voltages, fermi_level_J, poisson_solver_func,
                    schrodinger_solver_config=sch_config,
                    initial_potential_V=None
                )
                if base_timings["converged"] and potential_base is not None:
                    perturbed_voltages = generate_perturbed_voltages(base_random_voltages)
                    timings_perturbed, _ = benchmark_sc_iteration(
                        perturbed_voltages, fermi_level_J, poisson_solver_func,
                        schrodinger_solver_config=sch_config,
                        initial_potential_V=potential_base
                    )
                    if timings_perturbed["converged"]:
                        perturbed_warm_start_run_timings.append(timings_perturbed)
                        converged_perturbed_warm +=1
                    # else:
                    #     print(f"    Sample {i_sample+1} (Perturbed Warm) for {sch_name}/{poisson_name} did not converge.")
                # else:
                #     print(f"    Sample {i_sample+1} (Base for Perturbed Warm) for {sch_name}/{poisson_name} did not converge. Skipping perturbed run.")
            current_config_stats["perturbed_warm_starts"] = calculate_stats(perturbed_warm_start_run_timings)
            print(f"    Converged {converged_perturbed_warm}/{N_SAMPLES} times for Perturbed Warm Starts (after base converged).")
            
            benchmark_summary_stats[sch_name][poisson_name] = current_config_stats
            if not cold_start_run_timings and not perturbed_warm_start_run_timings: # Handle cases where a solver fails completely
                 if poisson_solver_methods[poisson_name] == solve_poisson_2d_fem_stub: # Check if it's the stub
                    print(f"    FEM Poisson solver is a stub and was skipped as expected.")


    # --- Print Summary of Results ---
    print("\n\n--- Benchmark Statistics Summary ---")
    for sch_name, poisson_results in benchmark_summary_stats.items():
        print(f"\nSchrödinger Solver Configuration: {sch_name}")
        for poisson_name, scenario_results in poisson_results.items():
            print(f"  Poisson Solver: {poisson_name}")
            for scenario_name, stats_dict in scenario_results.items():
                print(f"    Scenario: {scenario_name.replace('_', ' ').title()}")
                if not stats_dict or stats_dict["iterations"]["count"] == 0 : # Check if any samples contributed
                    print(f"      No successful runs to report statistics.")
                    if stats_dict and "converged_count" in stats_dict["iterations"]:
                         print(f"      (Converged {stats_dict['iterations']['converged_count']}/{N_SAMPLES} samples)")
                    continue
                
                conv_count = stats_dict["iterations"]["converged_count"]
                print(f"      Converged Samples: {conv_count}/{N_SAMPLES}")
                if conv_count > 0:
                    for metric, values in stats_dict.items():
                        if "converged_count" in values: # Already printed
                            pass
                        print(f"      Avg {metric.replace('_', ' ').title()}: {values['mean']:.3f} s (std: {values['std']:.3f})")
                        if metric == "iterations": # Iterations is not time
                             print(f"      Avg {metric.replace('_', ' ').title()}: {values['mean']:.1f} (std: {values['std']:.1f})")


    print("\n--- Notes ---")
    print(f"1. Statistics are based on {N_SAMPLES} random voltage samples per scenario.")
    print("2. 'Avg Time' is per self-consistent iteration for schrodinger_time_avg, charge_calc_time_avg, poisson_time_avg.")
    print("3. Spectral Poisson method uses periodic BCs; Finite Difference uses Dirichlet phi=0.")
    print("4. For more detailed profiling of specific functions, consider using cProfile:")
    print("   Example: python -m cProfile -s cumtime benchmark_solver_schemes.py")
    print("5. To implement FEM: replace 'solve_poisson_2d_fem_stub' with a working FEM solver.")
    print("6. Schrödinger solver 'tol' and 'maxiter' can be tuned within 'schrodinger_solver_configs_to_test'.")

    print("\nBenchmark script finished.")
