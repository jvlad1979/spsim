#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Schrödinger-Poisson simulator for a semiconductor quantum dot device.
Modified to simulate charge stability diagrams by sweeping two gate voltages.
"""

import numpy as np
import matplotlib.pyplot as plt
# matplotlib.patches was not used.
import time
import os

# Imports from spsim package
from spsim import constants
from spsim.device.potential import get_external_potential
from spsim.simulation_runtime.charge_density import calculate_total_electrons
from spsim.simulation_runtime.sc_solver import self_consistent_solver_2d
from spsim.measurement_helpers.sweep_utils import get_hilbert_order


# --- Physical Constants ---
hbar = constants.hbar
m_e = constants.m_e
e = constants.e  # Elementary charge
epsilon_0 = constants.epsilon_0

# --- Material Parameters (e.g., GaAs) ---
# These are now taken directly from spsim.constants where they are pre-calculated or defined
m_eff = constants.m_eff  # Effective mass
eps_r = constants.eps_r  # Relative permittivity
epsilon = constants.epsilon # Permittivity of material

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
# The K-space grid (Kx, Ky, K_sq) is no longer needed globally,
# as the spsim.solvers.poisson.solve_poisson_2d_spectral handles it internally.

# --- Device Parameters ---
# The get_external_potential function is now imported from spsim.device.potential.
# Gate definitions are part of the library function.

# --- Schrödinger Solver (2D) ---
# The solve_schrodinger_2d function is now imported from spsim.solvers.schrodinger.
# (It's used internally by the self_consistent_solver_2d from spsim)

# --- Charge Density Calculation (2D) ---
# The calculate_charge_density_2d function is used by self_consistent_solver_2d from spsim.

# --- Total Electron Number Calculation ---
# The calculate_total_electrons function is now imported from spsim.simulation_runtime.charge_density.

# --- Poisson Solver (2D) ---
# The solve_poisson_2d (finite difference) and solve_poisson_2d_spectral functions
# are now imported from spsim.solvers.poisson.
# (They are used internally by the self_consistent_solver_2d from spsim)

# --- Hilbert Curve Ordering ---
# The get_hilbert_order function is now imported from spsim.measurement_helpers.sweep_utils.

# --- Self-Consistent Iteration (2D) ---
# The self_consistent_solver_2d function is now imported from spsim.simulation_runtime.sc_solver.


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
    # Call to spsim's get_external_potential requires Lx, Ly
    initial_ext_pot_J = get_external_potential(X, Y, ref_voltages, Lx, Ly)
    # Set Fermi level slightly above the potential minimum of the reference configuration
    # to ensure some states are occupied within the sweep range.
    fermi_level_J = (
        np.min(initial_ext_pot_J) + 0.05 * e
    )  # Example: 50 meV above min potential

    print("\n--- Starting Charge Stability Diagram Simulation (using spsim package) ---")
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

            # Run the self-consistent solver using spsim.simulation_runtime.sc_solver
            sc_results = self_consistent_solver_2d(
                voltages=current_voltages,
                fermi_level=fermi_level_J,
                Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, # Grid parameters
                max_iter=30,  # From original script's call
                tol=1e-4,    # From original script's call
                mixing=0.1,  # From original script's call
                verbose=False, # From original script's call
                initial_potential_V=warm_start_potential,
                poisson_solver_type="finite_difference", # From original script's call
                schrodinger_solver_config=None # Use default spsim Schrödinger solver settings
            )

            # sc_results is (total_potential_J, charge_density, eigenvalues, eigenvectors_2d)
            # or (None, None, None, None) on failure
            if sc_results[0] is not None: # Check if total_potential_J is not None
                total_potential_J, final_charge_density, _eigenvalues, _eigenvectors = sc_results
                
                # Call to spsim's calculate_total_electrons requires dx, dy
                total_electrons = calculate_total_electrons(final_charge_density, dx, dy)
                total_electron_map[i, j] = total_electrons # Store using original grid indices

                # For warm start: calculate electrostatic potential from total and external
                current_external_potential_J = get_external_potential(X, Y, current_voltages, Lx, Ly)
                converged_electrostatic_potential_V = (current_external_potential_J - total_potential_J) / e
                
                converged_potentials_map[i][j] = converged_electrostatic_potential_V
                potential_from_previous_point = converged_electrostatic_potential_V # Update for next Hilbert point
                print(f"  -> Total Electrons: {total_electrons:.3f}")
            else:
                print(
                    f"  -> Simulation failed for point ({i + 1},{j + 1}) using spsim. Storing NaN."
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

                # Run the self-consistent solver using spsim.simulation_runtime.sc_solver
                sc_results = self_consistent_solver_2d(
                    voltages=current_voltages,
                    fermi_level=fermi_level_J,
                    Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy, # Grid parameters
                    max_iter=30,  # From original script's call
                    tol=1e-4,    # From original script's call
                    mixing=0.1,  # From original script's call
                    verbose=False, # From original script's call
                    initial_potential_V=potential_from_previous_point_in_row,
                    poisson_solver_type="finite_difference", # From original script's call
                    schrodinger_solver_config=None # Use default spsim Schrödinger solver settings
                )

                if sc_results[0] is not None: # Check if total_potential_J is not None
                    total_potential_J, final_charge_density, _eigenvalues, _eigenvectors = sc_results

                    # Call to spsim's calculate_total_electrons requires dx, dy
                    total_electrons = calculate_total_electrons(final_charge_density, dx, dy)
                    total_electron_map[i, j] = total_electrons
                    
                    # For warm start: calculate electrostatic potential from total and external
                    current_external_potential_J = get_external_potential(X, Y, current_voltages, Lx, Ly)
                    converged_electrostatic_potential_V = (current_external_potential_J - total_potential_J) / e

                    print(f"  -> Total Electrons: {total_electrons:.3f}")
                    potential_from_previous_point_in_row = converged_electrostatic_potential_V # Update for next point in row
                else:
                    print(
                        f"  -> Simulation failed for point ({i + 1},{j + 1}) using spsim. Storing NaN."
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
