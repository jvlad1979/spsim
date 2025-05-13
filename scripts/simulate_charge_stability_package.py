
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import scipy.constants as const # Still useful for eV conversion

# Import modules from the quantum_dot_sim package
from quantum_dot_sim.constants import e # Elementary charge
from quantum_dot_sim.simulation_runtime import self_consistent_solver_2d, calculate_total_electrons
from quantum_dot_sim.measurement_helpers import get_hilbert_order

# --- Simulation Parameters ---
# Grid parameters
Lx = 150e-9  # Domain size in x (m)
Ly = 100e-9  # Domain size in y (m)
Nx = 60      # Number of grid points in x
Ny = 40      # Number of grid points in y
dx = Lx / (Nx - 1) # Grid spacing in x (m)
dy = Ly / (Ny - 1) # Grid spacing in y (m)

# Fermi level (relative to the conduction band minimum in the 2DEG)
# A typical value might be around 0.1 eV below the 2DEG CBM,
# which is often taken as the zero reference for potential.
# So, E_F = -0.1 eV = -0.1 * e Joules
fermi_level_eV = -0.1
fermi_level_J = fermi_level_eV * e # Fermi level in Joules

# Gate voltage sweep parameters
gate1_name = "P1"
gate2_name = "P2"
gate1_start_V = -0.5
gate1_stop_V = -0.1
gate1_steps = 20

gate2_start_V = -0.5
gate2_stop_V = -0.1
gate2_steps = 20

# Other gate voltages (fixed during the sweep)
fixed_voltages = {
    "B1": 0.0,
    "B2": 0.1, # Center barrier often positive to define two dots
    "B3": 0.0,
}

# Self-consistent solver parameters
max_sc_iter = 50
sc_tol = 1e-5
sc_mixing = 0.1
poisson_solver_type = "finite_difference" # Options: 'finite_difference', 'spectral', 'fem_stub'

# Schrödinger solver configuration (optional)
# Example: Use eigsh with 'SM' (smallest magnitude)
schrodinger_solver_config = {"method": "eigsh", "params": {"which": "SM", "k": 10}} # Find lowest 10 states

# Sweep strategy: 'row_by_row' or 'hilbert'
sweep_strategy = "hilbert" # 'row_by_row' or 'hilbert'

# Output directory
output_dir = "charge_stability_output"
os.makedirs(output_dir, exist_ok=True)

# --- Setup Sweep ---
gate1_values = np.linspace(gate1_start_V, gate1_stop_V, gate1_steps)
gate2_values = np.linspace(gate2_start_V, gate2_stop_V, gate2_steps)

# Create a list of all voltage points to simulate
voltage_points = []
for v1 in gate1_values:
    for v2 in gate2_values:
        voltages = fixed_voltages.copy()
        voltages[gate1_name] = v1
        voltages[gate2_name] = v2
        voltage_points.append(voltages)

# Determine sweep order
if sweep_strategy == "row_by_row":
    # Default order is already row-by-row (v1 changes outer, v2 changes inner)
    sweep_order_indices = list(range(len(voltage_points)))
    print("Using row-by-row sweep strategy.")
elif sweep_strategy == "hilbert":
    # Get Hilbert order for the grid of voltage points (gate1_steps x gate2_steps)
    # The get_hilbert_order function returns (i, j) indices for a grid.
    # We need to map these (i, j) indices back to the flattened voltage_points list.
    # The flattened list is ordered such that index k corresponds to
    # gate1_values[k // gate2_steps] and gate2_values[k % gate2_steps].
    # So, for a point (i, j) in the gate1_steps x gate2_steps grid,
    # the flattened index is i * gate2_steps + j.
    hilbert_indices_2d = get_hilbert_order(gate1_steps, gate2_steps)
    sweep_order_indices = [i * gate2_steps + j for i, j in hilbert_indices_2d]
    print("Using Hilbert curve sweep strategy.")
else:
    raise ValueError(f"Unknown sweep_strategy: {sweep_strategy}")

# Initialize storage for results
electron_map = np.zeros((gate1_steps, gate2_steps))
# Store the converged potential for warm starts
last_converged_potential_V = None # Initialize for cold start at the first point

# --- Run Simulation Sweep ---
print(f"Starting charge stability sweep ({gate1_steps}x{gate2_steps} points)...")
total_start_time = time.time()

for k, flat_idx in enumerate(sweep_order_indices):
    current_voltages = voltage_points[flat_idx]

    # Map flat index back to 2D grid indices (i, j) for storing results
    i = flat_idx // gate2_steps # Index for gate1_values
    j = flat_idx % gate2_steps  # Index for gate2_values

    v1 = current_voltages[gate1_name]
    v2 = current_voltages[gate2_name]

    print(f"\nSimulating point ({i+1}/{gate1_steps}, {j+1}/{gate2_steps}): {gate1_name}={v1:.3f}V, {gate2_name}={v2:.3f}V")

    # Run the self-consistent solver
    # Pass the last converged potential for warm start
    (
        total_potential_J,
        charge_density,
        eigenvalues,
        eigenvectors_2d,
    ) = self_consistent_solver_2d(
        current_voltages,
        fermi_level_J,
        Nx, Ny, Lx, Ly, dx, dy,
        max_iter=max_sc_iter,
        tol=sc_tol,
        mixing=sc_mixing,
        poisson_solver_type=poisson_solver_type,
        schrodinger_solver_config=schrodinger_solver_config,
        initial_potential_V=last_converged_potential_V, # Use warm start
        verbose=False # Keep SC solver quiet during sweep
    )

    if total_potential_J is not None:
        # Calculation was successful
        # Calculate total number of electrons
        total_electrons = calculate_total_electrons(charge_density, dx, dy)
        electron_map[i, j] = total_electrons

        # Store the converged electrostatic potential for the next iteration's warm start
        # total_potential_J = external_potential_J - e * electrostatic_potential_V
        # We need electrostatic_potential_V, which is not directly returned by default.
        # A better approach is to modify self_consistent_solver_2d to return it,
        # or recalculate it here if needed for warm start.
        # For now, let's assume the solver returns the total potential,
        # and we might need to adjust the solver return or warm start logic.
        # Let's assume for warm start we need the electrostatic part:
        # phi = (V_ext - V_total) / e
        # This requires V_ext, which is calculated inside the solver.
        # A simpler warm start is to pass the *total* potential, but the solver expects *electrostatic*.
        # Let's update the solver to return electrostatic_potential_V.
        # For now, we'll use a cold start for each point until the solver is updated.
        # TODO: Update self_consistent_solver_2d to return electrostatic_potential_V for proper warm start.
        # For now, disable warm start to avoid using potentially incorrect potential.
        last_converged_potential_V = None # Force cold start for now

        print(f"  Total electrons: {total_electrons:.2f}")
    else:
        # Calculation failed for this point
        print("  Calculation failed.")
        electron_map[i, j] = np.nan # Mark as failed

total_end_time = time.time()
print(f"\nSweep finished in {total_end_time - total_start_time:.2f} seconds.")

# --- Save Results ---
filename_npz = os.path.join(output_dir, "charge_stability_data.npz")
np.savez(
    filename_npz,
    gate1_name=gate1_name,
    gate2_name=gate2_name,
    gate1_values=gate1_values,
    gate2_values=gate2_values,
    electron_map=electron_map,
    fermi_level_eV=fermi_level_eV,
    Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
    max_sc_iter=max_sc_iter,
    sc_tol=sc_tol,
    sc_mixing=sc_mixing,
    poisson_solver_type=poisson_solver_type,
    schrodinger_solver_config=schrodinger_solver_config,
    sweep_strategy=sweep_strategy,
    fixed_voltages=fixed_voltages # Save fixed voltages for context
)
print(f"Results saved to {filename_npz}")

# --- Plot Results ---
print("Plotting results...")

# Plot 1: Raw electron number map
plt.figure(figsize=(8, 6))
extent = [gate1_values.min(), gate1_values.max(), gate2_values.min(), gate2_values.max()]
plt.imshow(
    electron_map.T, # Transpose to match (V1, V2) -> (x, y) mapping
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Total Electrons")
plt.xlabel(f"{gate1_name} Voltage (V)")
plt.ylabel(f"{gate2_name} Voltage (V)")
plt.title("Charge Stability Diagram (Raw Electron Number)")
plt.grid(True, linestyle="--", alpha=0.5)
filename_raw_plot = os.path.join(output_dir, "charge_stability_raw.png")
plt.savefig(filename_raw_plot)
print(f"Raw electron map plot saved to {filename_raw_plot}")

# Plot 2: Rounded electron number map (emphasize plateaus)
plt.figure(figsize=(8, 6))
plt.imshow(
    np.round(electron_map).T, # Transpose and round
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    interpolation="nearest" # Use nearest interpolation for blocky appearance
)
plt.colorbar(label="Rounded Electron Number")
plt.xlabel(f"{gate1_name} Voltage (V)")
plt.ylabel(f"{gate2_name} Voltage (V)")
plt.title("Charge Stability Diagram (Rounded Electron Number)")
plt.grid(True, linestyle="--", alpha=0.5)
filename_rounded_plot = os.path.join(output_dir, "charge_stability_rounded.png")
plt.savefig(filename_rounded_plot)
print(f"Rounded electron map plot saved to {filename_rounded_plot}")


# Optional: Plot sweep order for Hilbert curve visualization
if sweep_strategy == "hilbert":
    plt.figure(figsize=(6, 6))
    # Map 2D indices (i, j) back to voltage values
    ordered_v1 = [gate1_values[idx[0]] for idx in hilbert_indices_2d]
    ordered_v2 = [gate2_values[idx[1]] for idx in hilbert_indices_2d]
    plt.plot(ordered_v1, ordered_v2, marker='o', linestyle='-', markersize=3, linewidth=0.5)
    plt.xlabel(f"{gate1_name} Voltage (V)")
    plt.ylabel(f"{gate2_name} Voltage (V)")
    plt.title(f"Hilbert Curve Sweep Order ({gate1_steps}x{gate2_steps})")
    plt.grid(True, linestyle="--", alpha=0.5)
    filename_hilbert_plot = os.path.join(output_dir, "hilbert_sweep_order.png")
    plt.savefig(filename_hilbert_plot)
    print(f"Hilbert sweep order plot saved to {filename_hilbert_plot}")


plt.show() # Display plots
