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
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For colormaps
# import cProfile # For more detailed profiling

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

# --- Simulation Grid (Reduced size for faster benchmarking) ---
Lx = 150e-9  # Length of the simulation domain in x (m)
Ly = 100e-9  # Length of the simulation domain in y (m)
Nx = 90  # Number of grid points in x (reduced from 75 for speed)
Ny = 60  # Number of grid points in y (reduced from 50 for speed)
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
Kx_grid, Ky_grid = np.meshgrid(kx, ky, indexing="ij")
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

    diagonals = [
        diag_terms,
        offdiag_x_terms[:-1],
        offdiag_x_terms[:-1],
        offdiag_y_terms[:-Nx],
        offdiag_y_terms[:-Nx],
    ]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx):  # Boundary condition adjustments for finite difference
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0

    H = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    method = solver_config.get("method", "eigsh")
    params = solver_config.get("params", {})
    num_eigenstates = params.get("k", 10)

    eigenvalues = np.array([])
    eigenvectors_flat = np.empty((N_total, 0))

    try:
        if method == "eigsh":
            eigsh_params = params.copy()
            if eigsh_params.pop("use_sigma_min_potential", False):
                sigma = np.min(potential_flat)
                # Ensure sigma is not exactly an eigenvalue, or too close to cause issues (optional refinement)
                # sigma -= 1e-9 * e # Small shift if needed
                eigsh_params["sigma"] = sigma
                if (
                    "which" not in eigsh_params
                ):  # sigma requires which='LM' or 'LA' typically
                    eigsh_params["which"] = "LM"

            # Remove non-eigsh specific params if any were added for other types
            eigsh_params.pop("use_random_X", None)

            eigenvalues, eigenvectors_flat = spla.eigsh(H, **eigsh_params)

        elif method == "lobpcg":
            lobpcg_params = params.copy()
            k_lobpcg = lobpcg_params.pop("k")  # k is handled by X shape

            if lobpcg_params.pop("use_random_X", False):
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
            if "tol" not in lobpcg_params:
                lobpcg_params["tol"] = 1e-5
            if "maxiter" not in lobpcg_params:
                lobpcg_params["maxiter"] = N_total // 2  # Heuristic

            eigenvalues, eigenvectors_flat = spla.lobpcg(
                H, X_init, largest=False, **lobpcg_params
            )
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
        if norm_sq > 1e-12:  # Avoid division by zero for zero vectors
            norm = np.sqrt(norm_sq)
            eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")
        else:
            eigenvectors[:, :, i] = psi_flat.reshape(
                (Nx, Ny), order="C"
            )  # Keep as is if norm is zero
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

    diagonals = [
        diag_terms,
        offdiag_x_terms[:-1],
        offdiag_x_terms[:-1],
        offdiag_y_terms[:-Nx],
        offdiag_y_terms[:-Nx],
    ]
    offsets = [0, -1, 1, -Nx, Nx]

    for i in range(1, Nx):  # Boundary condition adjustments
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
        if info != 0:
            phi_flat = np.zeros_like(rho_flat)
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
    if K_sq_solver[0, 0] == 0:  # Should be true for standard fftfreq setup
        K_sq_solver[0, 0] = 1.0  # Avoid division by zero, phi_k[0,0] is set to 0 anyway

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
def benchmark_sc_iteration(
    voltages,
    fermi_level,
    poisson_solver_func,
    schrodinger_solver_config,
    initial_potential_V=None,
    max_iter=30,
    tol=1e-4,
    mixing=0.1,
):
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
        "converged": False,
    }

    loop_start_time = time.time()

    is_warm_start = initial_potential_V is not None
    # Initial guess for electrostatic potential (Volts)
    if not is_warm_start:
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
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(
            total_potential_J, schrodinger_solver_config
        )
        timings["schrodinger_times_per_iter"].append(time.time() - sch_start_time)
        if (
            not eigenvalues.size or eigenvectors_2d.shape[2] == 0
        ):  # Check if any eigenvectors were returned
            print("Error in Schrödinger solver during SC iteration. Aborting.")
            timings["total_time"] = time.time() - loop_start_time
            return timings, None  # Indicate failure

        # 2. Calculate charge density
        cc_start_time = time.time()
        new_charge_density = calculate_charge_density_2d(
            eigenvalues, eigenvectors_2d, fermi_level
        )
        timings["charge_calc_times_per_iter"].append(time.time() - cc_start_time)

        # 3. Solve Poisson equation
        ps_start_time = time.time()
        new_electrostatic_potential_V = poisson_solver_func(new_charge_density)
        timings["poisson_times_per_iter"].append(time.time() - ps_start_time)

        # 4. Check for convergence
        potential_diff_norm = np.linalg.norm(
            new_electrostatic_potential_V - electrostatic_potential_V
        ) * np.sqrt(dx * dy)
        if potential_diff_norm < tol:
            timings["converged"] = True
            electrostatic_potential_V = new_electrostatic_potential_V  # Final update
            break

        # 5. Mix potential for stability
        current_iter_mixing = mixing
        if is_warm_start and i == 0:  # First iteration of a warm start
            current_iter_mixing = 0.5  # Use a more aggressive mixing for the first step
            # print(f"  Warm start: Using aggressive mixing {current_iter_mixing} for iter 0") # Optional debug

        electrostatic_potential_V += current_iter_mixing * (
            new_electrostatic_potential_V - electrostatic_potential_V
        )
    else:  # Loop finished without break (no convergence)
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

    N_SAMPLES = 100  # Number of random voltage samples
    NUM_EIGENSTATES = 10  # Default number of eigenstates to solve for

    # Define voltage ranges for random generation, aligned with charge stability simulations
    # Gate names: "P1", "P2", "B1", "B2", "B3"
    # P1, P2 swept in (-0.1, 0.0) in stability. B1=0.15, B2=0.20, B3=0.15.
    voltage_config = {
        "P1": {
            "range": (-0.1, 0.0),
            "perturb_std": 0.02,
        },  # Typical plunger sweep range
        "P2": {
            "range": (-0.1, 0.0),
            "perturb_std": 0.01,
        },  # Typical plunger sweep range
        "B1": {"range": (0.1, 0.2), "perturb_std": 0.01},  # Around fixed 0.15V
        "B2": {"range": (0.15, 0.25), "perturb_std": 0.01},  # Around fixed 0.20V
        "B3": {"range": (0.1, 0.2), "perturb_std": 0.01},  # Around fixed 0.15V
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
    fermi_level_J = (
        np.min(_ext_pot_avg) + 0.02 * e
    )  # 20 meV above min of average potential

    schrodinger_solver_configs_to_test = [
        {
            "name": "eigsh_SM",
            "method": "eigsh",
            "params": {"k": NUM_EIGENSTATES, "which": "SM"},
        },
        {
            "name": "eigsh_LM_sigma_min_pot",
            "method": "eigsh",
            "params": {
                "k": NUM_EIGENSTATES,
                "which": "LM",
                "use_sigma_min_potential": True,
                "tol": 1e-8,
            },
        },  # Added tol for sigma mode
        # LOBPCG can be sensitive; ensure params are well-chosen or add later.
        # {"name": "lobpcg_randX", "method": "lobpcg",
        #  "params": {"k": NUM_EIGENSTATES, "use_random_X": True, "tol": 1e-7, "maxiter": (Nx*Ny)//4}},
    ]

    # Add LOBPCG to the list of solver configurations to test
    if {
        "name": "lobpcg_randX",
        "method": "lobpcg",
        "params": {
            "k": NUM_EIGENSTATES,
            "use_random_X": True,
            "tol": 1e-7,
            "maxiter": (Nx * Ny) // 4,
        },
    } not in schrodinger_solver_configs_to_test:
        schrodinger_solver_configs_to_test.append(
            {
                "name": "lobpcg_randX",
                "method": "lobpcg",
                "params": {
                    "k": NUM_EIGENSTATES,
                    "use_random_X": True,
                    "tol": 1e-7,
                    "maxiter": (Nx * Ny) // 4,
                },
            }
        )

    poisson_solver_methods = {
        "Finite Difference": solve_poisson_2d_fd,
        "Spectral (FFT)": solve_poisson_2d_spectral,
    }

    benchmark_summary_stats = {}

    def calculate_stats(timings_list):
        if not timings_list:
            # Ensure all expected keys are present even for empty lists
            # Removed "iterations" from stat_keys
            stat_keys = [
                "total_time",
                "ext_potential_time",
                "schrodinger_time_avg",
                "charge_calc_time_avg",
                "poisson_time_avg",
            ]
            # Initialize converged_count for the "iterations" key as well, as it's used for reporting total converged samples
            base_return = {
                key: {"mean": np.nan, "std": np.nan, "count": 0, "converged_count": 0}
                for key in stat_keys
            }
            base_return["iterations"] = {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
                "converged_count": 0,
            }  # Keep structure for converged_count
            return base_return

        stats = {}
        # converged_count is based on the number of items in timings_list, assuming only converged runs are added
        converged_runs = len(timings_list)

        # Define keys for which to calculate full stats (mean, std)
        keys_for_full_stats = [
            "total_time",
            "ext_potential_time",
            "schrodinger_time_avg",
            "charge_calc_time_avg",
            "poisson_time_avg",
        ]

        all_timing_keys = list(
            timings_list[0].keys()
        )  # Get all keys from a sample timing dict

        for key in all_timing_keys:
            if (
                key.endswith("_per_iter") or key == "converged"
            ):  # Skip these raw lists or boolean
                continue

            if key in keys_for_full_stats:
                values = [
                    d[key] for d in timings_list if isinstance(d.get(key), (int, float))
                ]
                if values:
                    stats[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                        "converged_count": converged_runs,
                    }
                else:
                    stats[key] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "count": 0,
                        "converged_count": converged_runs,
                    }
            elif (
                key == "iterations"
            ):  # For "iterations", only store count and converged_count
                values = [
                    d[key] for d in timings_list if isinstance(d.get(key), (int, float))
                ]  # Still get values for count
                stats[key] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "count": len(values),
                    "converged_count": converged_runs,
                }
            # else: # Other keys not explicitly handled
            # pass
        return stats

    for sch_config in schrodinger_solver_configs_to_test:
        sch_name = sch_config["name"]
        benchmark_summary_stats[sch_name] = {}
        for poisson_name, poisson_solver_func in poisson_solver_methods.items():
            print(
                f"\n--- Benchmarking Schrödinger: {sch_name} | Poisson: {poisson_name} ---"
            )
            current_config_stats = {}

            # --- Scenario 1: Random Voltages (Cold Starts) ---
            print(f"  Running Scenario: {N_SAMPLES} Random Cold Starts")
            cold_start_run_timings = []
            for i_sample in range(N_SAMPLES):
                random_voltages = generate_random_voltages()
                timings, _ = benchmark_sc_iteration(
                    random_voltages,
                    fermi_level_J,
                    poisson_solver_func,
                    schrodinger_solver_config=sch_config,
                    initial_potential_V=None,
                )
                if timings["converged"]:
                    cold_start_run_timings.append(timings)
                # else:
                #     print(f"    Sample {i_sample+1} (Cold) for {sch_name}/{poisson_name} did not converge.")
            current_config_stats["random_cold_starts"] = calculate_stats(
                cold_start_run_timings
            )
            num_converged_cold = len(cold_start_run_timings)
            print(
                f"    Converged {num_converged_cold}/{N_SAMPLES} times for Random Cold Starts."
            )

            # --- Scenario 2: Perturbed Warm Starts ---
            print(f"  Running Scenario: {N_SAMPLES} Perturbed Warm Starts")
            perturbed_warm_start_run_timings = []
            converged_perturbed_warm = 0
            for i_sample in range(N_SAMPLES):
                base_random_voltages = generate_random_voltages()
                base_timings, potential_base = benchmark_sc_iteration(
                    base_random_voltages,
                    fermi_level_J,
                    poisson_solver_func,
                    schrodinger_solver_config=sch_config,
                    initial_potential_V=None,
                )
                if base_timings["converged"] and potential_base is not None:
                    perturbed_voltages = generate_perturbed_voltages(
                        base_random_voltages
                    )
                    timings_perturbed, _ = benchmark_sc_iteration(
                        perturbed_voltages,
                        fermi_level_J,
                        poisson_solver_func,
                        schrodinger_solver_config=sch_config,
                        initial_potential_V=potential_base,
                    )
                    if timings_perturbed["converged"]:
                        perturbed_warm_start_run_timings.append(timings_perturbed)
                        converged_perturbed_warm += 1
                    # else:
                    #     print(f"    Sample {i_sample+1} (Perturbed Warm) for {sch_name}/{poisson_name} did not converge.")
                # else:
                #     print(f"    Sample {i_sample+1} (Base for Perturbed Warm) for {sch_name}/{poisson_name} did not converge. Skipping perturbed run.")
            current_config_stats["perturbed_warm_starts"] = calculate_stats(
                perturbed_warm_start_run_timings
            )
            print(
                f"    Converged {converged_perturbed_warm}/{N_SAMPLES} times for Perturbed Warm Starts (after base converged)."
            )

            benchmark_summary_stats[sch_name][poisson_name] = current_config_stats
            if (
                not cold_start_run_timings and not perturbed_warm_start_run_timings
            ):  # Handle cases where a solver fails completely
                if (
                    poisson_solver_methods[poisson_name] == solve_poisson_2d_fem_stub
                ):  # Check if it's the stub
                    print(
                        f"    FEM Poisson solver is a stub and was skipped as expected."
                    )

    # --- Print Summary of Results ---
    print("\n\n--- Benchmark Statistics Summary ---")
    for sch_name, poisson_results in benchmark_summary_stats.items():
        print(f"\nSchrödinger Solver Configuration: {sch_name}")
        for poisson_name, scenario_results in poisson_results.items():
            print(f"  Poisson Solver: {poisson_name}")
            for scenario_name, stats_dict in scenario_results.items():
                print(f"    Scenario: {scenario_name.replace('_', ' ').title()}")
                if (
                    not stats_dict or stats_dict["iterations"]["count"] == 0
                ):  # Check if any samples contributed
                    print(f"      No successful runs to report statistics.")
                    if stats_dict and "converged_count" in stats_dict["iterations"]:
                        print(
                            f"      (Converged {stats_dict['iterations']['converged_count']}/{N_SAMPLES} samples)"
                        )
                    continue

                conv_count = stats_dict["iterations"]["converged_count"]
                print(f"      Converged Samples: {conv_count}/{N_SAMPLES}")
                if conv_count > 0:
                    for metric, values in stats_dict.items():
                        if (
                            "converged_count" in values and metric != "iterations"
                        ):  # Already printed for converged samples
                            pass
                        if metric == "iterations":  # Skip printing stats for iterations
                            continue
                        print(
                            f"      Avg {metric.replace('_', ' ').title()}: {values['mean']:.3f} s (std: {values['std']:.3f})"
                        )

    print("\n--- Notes ---")
    print(
        f"1. Statistics are based on {N_SAMPLES} random voltage samples per scenario."
    )
    print(
        "2. 'Avg Time' is per self-consistent iteration for schrodinger_time_avg, charge_calc_time_avg, poisson_time_avg."
    )
    print(
        "3. Spectral Poisson method uses periodic BCs; Finite Difference uses Dirichlet phi=0."
    )
    print(
        "4. For more detailed profiling of specific functions, consider using cProfile:"
    )
    print("   Example: python -m cProfile -s cumtime benchmark_solver_schemes.py")
    print(
        "5. To implement FEM: replace 'solve_poisson_2d_fem_stub' with a working FEM solver."
    )
    print(
        "6. Schrödinger solver 'tol' and 'maxiter' can be tuned within 'schrodinger_solver_configs_to_test'."
    )

    # --- Plotting Summary ---
    def plot_benchmark_summary(summary_stats, n_samples_total):
        """Plots a summary of the benchmark statistics."""
        schrodinger_configs = list(summary_stats.keys())
        if not schrodinger_configs:
            print("No data to plot.")
            return

        poisson_methods = list(summary_stats[schrodinger_configs[0]].keys())
        scenarios = list(
            summary_stats[schrodinger_configs[0]][poisson_methods[0]].keys()
        )

        metrics_to_plot = {
            "total_time": "Avg Total Time (s)",
            "schrodinger_time_avg": "Avg Schrödinger Time / Iter (s)",
            "poisson_time_avg": "Avg Poisson Time / Iter (s)",
            # "iterations": "Avg Iterations" # Removed iterations from plot
        }

        n_metrics = len(metrics_to_plot)
        n_scenarios = len(scenarios)

        # One figure per scenario for clarity
        for scenario_idx, scenario_name in enumerate(scenarios):
            fig, axs = plt.subplots(n_metrics, 1, figsize=(8, 12), sharex=True)
            if n_metrics == 1:  # Make axs iterable if only one metric
                axs = [axs]
            fig.suptitle(
                f"Benchmark Results: {scenario_name.replace('_', ' ').title()}",
                fontsize=16,
            )

            bar_width = 0.15

            for metric_idx, (metric_key, metric_label) in enumerate(
                metrics_to_plot.items()
            ):
                ax = axs[metric_idx]

                x_labels = schrodinger_configs
                x_pos = np.arange(len(x_labels))

                for i, poisson_name in enumerate(poisson_methods):
                    means = []
                    stds = []
                    converged_counts = []
                    for sch_name in schrodinger_configs:
                        try:
                            data = summary_stats[sch_name][poisson_name][scenario_name][
                                metric_key
                            ]
                            means.append(data["mean"])
                            stds.append(data["std"])
                            converged_counts.append(
                                summary_stats[sch_name][poisson_name][scenario_name][
                                    "iterations"
                                ]["converged_count"]
                            )
                        except (
                            KeyError,
                            TypeError,
                        ):  # Handle missing data or structure issues
                            means.append(0)  # Plot as zero if data missing
                            stds.append(0)
                            converged_counts.append(0)

                    rects = ax.bar(
                        x_pos
                        + i * bar_width
                        - bar_width / 2 * (len(poisson_methods) - 1),
                        means,
                        bar_width,
                        yerr=stds,
                        label=f"{poisson_name}",
                        capsize=5,
                    )

                    # Add text for converged count on top of bars
                    for rect_idx, rect in enumerate(rects):
                        height = rect.get_height()
                        y_val = height + stds[rect_idx]  # Position above error bar
                        conv_count = converged_counts[rect_idx]
                        if conv_count > 0 and not np.isnan(
                            height
                        ):  # Only show if bar exists and count > 0
                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                y_val
                                + 0.01
                                * np.nanmax(means),  # Adjust offset based on max value
                                f"{conv_count}/{n_samples_total}",
                                ha="center",
                                va="bottom",
                                fontsize=12,
                                rotation=0,
                            )

                ax.set_ylabel(metric_label)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=15, ha="right")
                ax.grid(True, axis="y", linestyle=":", alpha=0.7)
                if metric_idx == 0:  # Add legend to the first subplot
                    ax.legend(
                        title="Poisson Solver", loc="upper left", bbox_to_anchor=(1, 1)
                    )

            plt.tight_layout(
                rect=[0, 0, 0.85, 0.96]
            )  # Adjust layout to make space for legend and suptitle
            plot_filename = f"benchmark_summary_{scenario_name}.png"
            plt.savefig(plot_filename)
            print(f"Benchmark plot saved to {plot_filename}")
            # plt.show() # Optionally show plot

    # --- Plotting Summary ---

    def plot_benchmark_summary_academic(
        summary_stats,
        n_samples_total,
        output_filename_base="benchmark_summary_academic",
    ):
        """
        Plots a summary of the benchmark statistics, optimized for academic publications.
        Generates a single figure with subplots for all scenarios and metrics.
        """
        schrodinger_configs = list(summary_stats.keys())
        if not schrodinger_configs:
            print("No Schrödinger configurations to plot.")
            return

        try:
            poisson_methods = list(summary_stats[schrodinger_configs[0]].keys())
            if not poisson_methods:
                print("No Poisson methods to plot.")
                return
            scenarios = list(
                summary_stats[schrodinger_configs[0]][poisson_methods[0]].keys()
            )
            if not scenarios:
                print("No scenarios to plot.")
                return
        except (KeyError, IndexError):
            print(
                "summary_stats has an unexpected structure. Cannot determine Poisson methods or scenarios."
            )
            return

        metrics_to_plot = {
            "total_time": "Avg Total Time (s)",
            "schrodinger_time_avg": "Avg Schrödinger Time / Iter (s)",
            "poisson_time_avg": "Avg Poisson Time / Iter (s)",
            # "iterations": "Avg Iterations" # Ensure 'iterations' data includes 'converged_count'
        }
        if not metrics_to_plot:
            print("No metrics defined for plotting.")
            return

        n_metrics = len(metrics_to_plot)
        n_scenarios = len(scenarios)

        # --- Plotting Style and Parameters ---
        plt.style.use("seaborn-v0_8-paper")  # Cleaner style for publications
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10
        plt.rcParams["xtick.labelsize"] = 9
        plt.rcParams["ytick.labelsize"] = 9
        plt.rcParams["legend.fontsize"] = 9
        plt.rcParams["figure.titlesize"] = 14
        # plt.rcParams['font.family'] = 'sans-serif'
        # plt.rcParams['font.sans-serif'] = ['Arial'] # Example: Specify font if needed

        # --- Figure Size Calculation ---
        # Aim for ~3.5-4 inches per subplot width, ~3 inches per subplot height
        subplot_base_width = 4.0
        subplot_base_height = 3.0
        MAX_FIG_WIDTH = 18  # Max width in inches
        MAX_FIG_HEIGHT = 24  # Max height in inches

        # Calculate figure width and height
        fig_width = min(MAX_FIG_WIDTH, subplot_base_width * n_metrics)
        # Adjust height for suptitle and legend (approx 1.5-2 inches)
        fig_height = min(MAX_FIG_HEIGHT, subplot_base_height * n_scenarios + 2.0)

        fig, axs = plt.subplots(
            n_scenarios,
            n_metrics,
            figsize=(fig_width, fig_height),
            sharex="col",  # Share x-axis per column of metrics
            sharey=False,  # Different metrics will have different y-scales
            squeeze=False,  # Always return a 2D array for axs
        )

        fig.suptitle(
            "Benchmark Performance Summary", y=0.98 if n_scenarios == 1 else 0.99
        )

        # --- Bar and Color Configuration ---
        num_poisson_methods = len(poisson_methods)
        group_total_width_allowance = (
            0.8  # How much of the x-tick space the group of bars occupies
        )
        bar_width = group_total_width_allowance / num_poisson_methods

        # Colorblind-friendly colors
        colors = cm.get_cmap("tab10", num_poisson_methods).colors
        # Optional: hatches for B&W printing
        # hatches = ['/', '\\', 'x', '.', '*', '+', 'O'] * (num_poisson_methods // 7 + 1)

        plot_handles_legend = {}  # For storing one handle per Poisson method for the figure legend

        for scenario_idx, scenario_name in enumerate(scenarios):
            for metric_idx, (metric_key, metric_label) in enumerate(
                metrics_to_plot.items()
            ):
                ax = axs[scenario_idx, metric_idx]
                x_labels = schrodinger_configs
                x_pos = np.arange(len(x_labels))

                for i, poisson_name in enumerate(poisson_methods):
                    means = []
                    stds = []
                    converged_counts = []
                    for sch_name in schrodinger_configs:
                        try:
                            data = summary_stats[sch_name][poisson_name][scenario_name][
                                metric_key
                            ]
                            means.append(data["mean"])
                            stds.append(data["std"])
                            # Assuming "iterations" key holds convergence data
                            converged_counts.append(
                                summary_stats[sch_name][poisson_name][scenario_name]
                                .get("iterations", {})  # Use .get for safety
                                .get("converged_count", 0)  # Default to 0 if not found
                            )
                        except (KeyError, TypeError):
                            means.append(0)
                            stds.append(0)
                            converged_counts.append(0)

                    # Corrected bar positioning: (i - (N-1)/2) * bar_width
                    # N = num_poisson_methods
                    # This centers the group of bars around x_pos
                    position_offset = (i - (num_poisson_methods - 1) / 2) * bar_width
                    bar_positions = x_pos + position_offset

                    rects = ax.bar(
                        bar_positions,
                        means,
                        bar_width,
                        yerr=stds,
                        label=poisson_name,
                        color=colors[i % len(colors)],
                        # hatch=hatches[i % len(hatches)] if hatches else None, # Optional hatch
                        capsize=4,
                        alpha=0.85,
                    )
                    if poisson_name not in plot_handles_legend:
                        plot_handles_legend[poisson_name] = rects[0]

                    # Add text for converged count on top of bars
                    for rect_idx, rect in enumerate(rects):
                        height = rect.get_height()
                        mean_val = means[rect_idx]
                        std_val = stds[rect_idx]

                        # Position text above the error bar or bar
                        y_val_text_base = (
                            mean_val + std_val if not np.isnan(std_val) else mean_val
                        )
                        if np.isnan(y_val_text_base):
                            y_val_text_base = 0  # Handle case where mean itself is NaN

                        conv_count = converged_counts[rect_idx]

                        # Only show text if bar has a non-NaN height and count > 0 (or if you want to show 0/N)
                        if (
                            not np.isnan(height) and n_samples_total > 0
                        ):  # Show even if conv_count is 0
                            # Dynamic offset based on y-axis scale to prevent overlap
                            y_axis_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            text_v_offset = y_axis_range * 0.02  # 2% of y-axis height

                            # Ensure text is above, check for negative bars if applicable
                            final_y_for_text = (
                                y_val_text_base + text_v_offset
                                if height >= 0
                                else y_val_text_base
                                - text_v_offset
                                - (y_axis_range * 0.05)
                            )

                            ax.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                final_y_for_text,
                                f"{conv_count}/{n_samples_total}",
                                ha="center",
                                va="bottom" if height >= 0 else "top",
                                fontsize=8,  # Slightly larger for readability
                                rotation=90,  # Keep 90 for compactness
                            )

                # --- Axis Labels, Ticks, and Titles for Subplot ---
                ax.set_ylabel(metric_label)
                ax.set_xticks(x_pos)

                if (
                    scenario_idx == n_scenarios - 1
                ):  # Only show x-labels on the bottom row
                    ax.set_xticklabels(
                        [label.replace("_", " ").title() for label in x_labels],
                        rotation=45,
                        ha="right",
                    )
                else:
                    ax.set_xticklabels([])

                ax.grid(True, axis="y", linestyle=":", alpha=0.7)

                # Set subplot title to indicate the scenario (and metric implicitly by column)
                if n_metrics > 1 and n_scenarios > 1:  # Full grid titles
                    ax.set_title(
                        f"{scenario_name.replace('_', ' ').title()}\n({metric_label.split('(')[0].strip()})",
                        fontsize=10,  # Smaller title for individual plots if many
                    )
                elif n_scenarios > 1:  # Only scenario in title if single metric column
                    ax.set_title(
                        f"{scenario_name.replace('_', ' ').title()}", fontsize=11
                    )
                elif (
                    n_metrics > 1
                ):  # Only metric in title if single scenario row (scenario in suptitle)
                    ax.set_title(f"{metric_label.split('(')[0].strip()}", fontsize=11)

                # Auto-adjust y-limits to give some padding, especially for text
                current_ylim = ax.get_ylim()
                padding_factor = 0.10  # 10% padding
                ylim_range = current_ylim[1] - current_ylim[0]
                if ylim_range == 0:  # Handle flat data
                    ylim_range = 1 if current_ylim[1] == 0 else abs(current_ylim[1])

                # Adjust top padding more if positive values, bottom if negative
                if all(
                    m >= 0 for m in means if not np.isnan(m)
                ):  # All positive or zero
                    ax.set_ylim(
                        current_ylim[0], current_ylim[1] + ylim_range * padding_factor
                    )
                elif all(
                    m <= 0 for m in means if not np.isnan(m)
                ):  # All negative or zero
                    ax.set_ylim(
                        current_ylim[0] - ylim_range * padding_factor, current_ylim[1]
                    )
                else:  # Mixed or general case
                    ax.set_ylim(
                        current_ylim[0] - ylim_range * (padding_factor / 2),
                        current_ylim[1] + ylim_range * (padding_factor / 2),
                    )

        # --- Figure Legend ---
        # Create handles and labels for the figure-level legend based on collected unique handles
        fig_legend_handles = [
            plot_handles_legend[pn]
            for pn in poisson_methods
            if pn in plot_handles_legend
        ]
        fig_legend_labels = [pn for pn in poisson_methods if pn in plot_handles_legend]

        if fig_legend_handles:
            fig.legend(
                fig_legend_handles,
                fig_legend_labels,
                title="Poisson Solver",
                loc="lower center",
                bbox_to_anchor=(
                    0.5,
                    0.01,
                ),  # Adjust 0.01 based on fig_height and font size
                ncol=min(len(poisson_methods), 4),  # Max 4 columns for legend
                frameon=True,
                fontsize=9,
            )

        # --- Layout Adjustments ---
        # Adjust layout to make space for legend and suptitle
        # The y value for suptitle and bbox_to_anchor for legend might need slight tweaks
        # depending on the final figure height and number of scenarios/legend items.
        bottom_padding = 0.15 if len(poisson_methods) > 0 else 0.05
        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.92 if n_scenarios > 1 else 0.88,
            bottom=bottom_padding,
            hspace=0.35,
            wspace=0.25,
        )

        # --- Save Figure ---
        png_filename = f"{output_filename_base}.png"
        pdf_filename = f"{output_filename_base}.pdf"

        plt.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"Academic benchmark plot saved to {png_filename}")
        try:
            plt.savefig(pdf_filename, dpi=300, bbox_inches="tight")
            print(f"Academic benchmark plot saved to {pdf_filename}")
        except Exception as e:
            print(f"Could not save PDF: {e}")

        # plt.show() # Optionally show plot interactively
        plt.close(fig)  # Close the figure to free memory
        # plt.show()  # Optionally show plot

    plot_benchmark_summary_academic(
        benchmark_summary_stats, N_SAMPLES, "my_benchmark_results"
    )

    print("\nBenchmark script finished.")
