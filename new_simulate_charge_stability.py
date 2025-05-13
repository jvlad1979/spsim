#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D Schrödinger-Poisson simulator for a semiconductor quantum dot device.
Modified to simulate charge stability diagrams by sweeping two gate voltages,
utilizing the spsim package.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os  # Added for creating output directory

# Import necessary functions and constants from spsim
from spsim.constants import e
from spsim.device.potential import get_external_potential
from spsim.solvers.schrodinger import solve_schrodinger_2d
from spsim.solvers.poisson import solve_poisson_2d_fd, solve_poisson_2d_spectral
from spsim.simulation_runtime.charge_density import calculate_charge_density_2d, calculate_total_electrons
from spsim.simulation_runtime.sc_solver import self_consistent_solver_2d
from spsim.measurement_helpers.sweep_utils import get_hilbert_order


# --- Simulation Grid Parameters (moved to main for clarity) ---
# These are now passed to the solver functions
Lx = 150e-9  # Length of the simulation domain in x (m)
Ly = 100e-9  # Length of the simulation domain in y (m)
Nx = 45  # Number of grid points in x (reduced)
Ny = 30  # Number of grid points in y (reduced)
N_total = Nx * Ny # Keep this for checks if needed

# --- Create coordinate arrays (moved here) ---
x_coords = np.linspace(0, Lx, Nx)
y_coords = np.linspace(0, Ly, Ny)
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]
X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")  # For potential


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

    # --- Use the imported get_external_potential function ---
    initial_ext_pot_J = get_external_potential(X, Y, ref_voltages, Lx, Ly)
    # Set Fermi level slightly above the potential minimum of the reference configuration
    # to ensure some states are occupied within the sweep range.
    fermi_level_J = (
        np.min(initial_ext_pot_J) + 0.05 * e
    )  # Example: 50 meV above min potential

    print("\n--- Starting Charge Stability Diagram Simulation (spsim) ---")
    print(
        f"Sweeping {gate1_name} from {gate1_voltages[0]:.3f} V to {gate1_voltages[-1]:.3f} V ({num_v1} points)"
    )
    print(
        f"Sweeping {gate2_name} from {gate2_voltages[0]:.3f} V to {gate2_voltages[-1]:.3f} V ({num_v2} points)"
    )
    print(f"Grid size: Nx={Nx}, Ny={Ny}")
    print(f"Fermi Level: {fermi_level_J / e:.4f} eV")
    print("-" * 50)

    output_dir = "stability_results_spsim"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in '{output_dir}'")

    sweep_start_time = time.time()
    total_points = num_v1 * num_v2
    completed_points = 0

    # --- Choose Sweep Strategy ---
    # Options: 'row_by_row', 'hilbert'
    sweep_strategy = "hilbert"  # Change this to 'row_by_row' for the original behavior

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
            warm_start_potential = potential_from_previous_point  # Default to previous Hilbert point

            # Check grid neighbor (i-1, j) if available and successful
            # This adds consideration of a direct grid neighbor
            if i > 0 and converged_potentials_map[i - 1][j] is not None:
                # print(f"  Using grid neighbor ({i-1},{j}) as warm start.") # Optional: uncomment for debugging
                warm_start_potential = converged_potentials_map[i - 1][j]
            # Could add check for (i, j-1) as well, but prioritizing one is simpler for now.

            # --- Use the imported self_consistent_solver_2d function ---
            final_charge_density, converged_potential_V = self_consistent_solver_2d(
                current_voltages,
                fermi_level_J,
                Nx, Ly, Lx, Ly, dx, dy, # Pass grid parameters
                max_iter=20,
                tol=5e-4,
                mixing=0.1,
                poisson_solver_type="finite_difference",
                initial_potential_V=warm_start_potential,  # Use the determined warm start
                verbose=False,
            )

            if final_charge_density is not None:
                # --- Use the imported calculate_total_electrons function ---
                total_electrons = calculate_total_electrons(final_charge_density, dx, dy)
                total_electron_map[i, j] = total_electrons  # Store using original grid indices
                converged_potentials_map[i][j] = converged_potential_V  # Store converged potential
                print(f"  -> Total Electrons: {total_electrons:.3f}")
                potential_from_previous_point = converged_potential_V  # Update for next Hilbert point
            else:
                print(
                    f"  -> Simulation failed for point ({i + 1},{j + 1}). Storing NaN."
                )
                total_electron_map[i, j] = np.nan
                converged_potentials_map[i][j] = None  # Store None for failed points
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
        potential_from_previous_point_in_row = None  # Specific to row strategy

        for i, v1 in enumerate(gate1_voltages):
            potential_from_previous_point_in_row = None  # Reset for each new row

            for j, v2 in enumerate(gate2_voltages):
                point_start_time = time.time()
                current_voltages = base_voltages.copy()
                current_voltages[gate1_name] = v1
                current_voltages[gate2_name] = v2

                print(
                    f"Running point ({i + 1}/{num_v1}, {j + 1}/{num_v2}): {gate1_name}={v1:.3f}V, {gate2_name}={v2:.3f}V"
                )

                # --- Use the imported self_consistent_solver_2d function ---
                final_charge_density, converged_potential_V = self_consistent_solver_2d(
                    current_voltages,
                    fermi_level_J,
                    Nx, Ny, Lx, Ly, dx, dy,  # Pass grid parameters
                    max_iter=20,
                    tol=5e-4,
                    mixing=0.1,
                    poisson_solver_type="finite_difference",
                    initial_potential_V=potential_from_previous_point_in_row,  # Use potential from previous point in row
                    verbose=False,
                )

                if final_charge_density is not None:
                    # --- Use the imported calculate_total_electrons function ---
                    total_electrons = calculate_total_electrons(final_charge_density, dx, dy)
                    total_electron_map[i, j] = total_electrons
                    print(f"  -> Total Electrons: {total_electrons:.3f}")
                    potential_from_previous_point_in_row = converged_potential_V  # Update for next point in row
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
