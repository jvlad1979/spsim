
import numpy as np
import time
from ..constants import e # Import elementary charge
from ..device.potential import get_external_potential # Import external potential function
from ..solvers.schrodinger import solve_schrodinger_2d # Import Schrödinger solver
from ..solvers.poisson import solve_poisson_2d_fd, solve_poisson_2d_spectral # Import Poisson solvers
from .charge_density import calculate_charge_density_2d # Import charge density calculation

def self_consistent_solver_2d(
    voltages,
    fermi_level,
    Nx, Ny, Lx, Ly, dx, dy, # Grid parameters
    max_iter=50,
    tol=1e-5,
    mixing=0.1,
    poisson_solver_type="finite_difference",
    schrodinger_solver_config=None, # Pass config to schrodinger solver
    initial_potential_V=None, # Warm start potential
    verbose=True, # Added verbose flag
):
    """
    Performs the self-consistent 2D Schrödinger-Poisson calculation.
    Allows choosing the Poisson solver ('finite_difference' or 'spectral').

    Args:
        voltages (dict): Dictionary mapping gate names (str) to applied voltages (float).
        fermi_level (float): Fermi level in Joules.
        Nx (int): Number of grid points in x.
        Ny (int): Number of grid points in y.
        Lx (float): Length of the simulation domain in x (m).
        Ly (float): Length of the simulation domain in y (m).
        dx (float): Grid spacing in x (m).
        dy (float): Grid spacing in y (m).
        max_iter (int): Maximum number of self-consistent iterations.
        tol (float): Convergence tolerance for the potential difference norm.
        mixing (float): Mixing parameter (0 to 1) for potential update.
        poisson_solver_type (str): Type of Poisson solver ('finite_difference' or 'spectral').
        schrodinger_solver_config (dict, optional): Configuration for the Schrödinger solver.
        initial_potential_V (np.ndarray, optional): Initial guess for the electrostatic potential (Volts) for warm start.
        verbose (bool): If True, print progress messages.

    Returns:
        tuple: (total_potential_J, charge_density, eigenvalues, eigenvectors_2d)
               total_potential_J (np.ndarray): Converged total potential energy in Joules.
               charge_density (np.ndarray): Converged charge density in C/m^2.
               eigenvalues (np.ndarray): Converged energy eigenvalues in Joules.
               eigenvectors_2d (np.ndarray): Converged wavefunctions (Nx, Ny, num_eigenstates).
               Returns (None, None, None, None) if calculation fails (e.g., Schrödinger solver error).
    """
    if verbose:
        print("Starting 2D self-consistent calculation...")
    start_time = time.time()

    # Initial guess for electrostatic potential (Volts)
    if initial_potential_V is None:
        electrostatic_potential_V = np.zeros((Nx, Ny))
        if verbose:
            print("  Using cold start (initial_potential_V is None).")
    else:
        # Ensure the warm start potential has the correct shape
        if initial_potential_V.shape != (Nx, Ny):
             print(f"Warning: Initial potential shape {initial_potential_V.shape} does not match grid shape ({Nx}, {Ny}). Using cold start.")
             electrostatic_potential_V = np.zeros((Nx, Ny))
        else:
            electrostatic_potential_V = initial_potential_V.copy()
            if verbose:
                print("  Using warm start (initial_potential_V is provided).")


    # Calculate external potential (once)
    # Need X, Y meshgrids for get_external_potential
    x_coords = np.linspace(0, Lx, Nx)
    y_coords = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    external_potential_J = get_external_potential(X, Y, voltages, Lx, Ly)  # Potential in Joules

    # Select Poisson solver function
    if poisson_solver_type == "finite_difference":
        poisson_solver_func = solve_poisson_2d_fd
    elif poisson_solver_type == "spectral":
        poisson_solver_func = solve_poisson_2d_spectral
    elif poisson_solver_type == "fem_stub":
         poisson_solver_func = solve_poisson_2d_fem_stub # Include stub option
    else:
        raise ValueError(f"Unknown poisson_solver_type: {poisson_solver_type}")

    # Adaptive mixing parameters (optional, can be added later if needed)
    # initial_mixing = mixing
    # min_mixing = 0.01
    # mixing_decay_rate = 0.9
    # previous_potential_diff_norm = float('inf')

    # Store the last successful results in case of non-convergence
    last_successful_results = (None, None, None, None)


    for i in range(max_iter):
        iter_start_time = time.time()
        if verbose:
            print(f"Iteration {i + 1}/{max_iter}")

        # 1. Calculate total potential energy V = V_ext + (-e * phi)
        total_potential_J = external_potential_J - e * electrostatic_potential_V

        # 2. Solve Schrödinger equation
        # Pass grid info and solver config to the solver
        eigenvalues, eigenvectors_2d = solve_schrodinger_2d(
            total_potential_J, Nx, Ny, dx, dy, solver_config=schrodinger_solver_config
        )
        if not eigenvalues.size or eigenvectors_2d.shape[2] == 0:
            print("Error in Schrödinger solver. Aborting SC loop.")
            # Return the last successful results if any, otherwise None
            return last_successful_results

        # 3. Calculate charge density
        new_charge_density = calculate_charge_density_2d(
            eigenvalues, eigenvectors_2d, fermi_level
        )

        # 4. Solve Poisson equation
        # Pass grid info to the solver
        new_electrostatic_potential_V = poisson_solver_func(new_charge_density, Nx, Ny, dx, dy)

        # 5. Check for convergence (using norm of potential difference)
        potential_diff_norm = np.linalg.norm(
            new_electrostatic_potential_V - electrostatic_potential_V
        ) * np.sqrt(dx * dy) # Use sqrt(dx*dy) for 2D norm scaling

        if verbose:
            print(f"  Potential difference norm: {potential_diff_norm:.3e}")

        # Store results from this iteration as potentially the last successful one
        last_successful_results = (
             total_potential_J,
             new_charge_density, # Use new_charge_density as it's based on current states
             eigenvalues,
             eigenvectors_2d,
        )


        if potential_diff_norm < tol:
            if verbose:
                print(f"Converged after {i + 1} iterations.")
            electrostatic_potential_V = new_electrostatic_potential_V # Final update
            break

        # 6. Mix potential for stability
        # Simple linear mixing
        electrostatic_potential_V = electrostatic_potential_V + mixing * (
            new_electrostatic_potential_V - electrostatic_potential_V
        )

        # Adaptive mixing logic could be added here based on potential_diff_norm
        # if i > 0 and potential_diff_norm > previous_potential_diff_norm:
        #     mixing *= mixing_decay_rate
        #     if mixing < min_mixing:
        #         mixing = min_mixing
        #     if verbose: print(f"  Reducing mixing parameter to {mixing:.3f}")
        # previous_potential_diff_norm = potential_diff_norm


        if verbose:
            iter_end_time = time.time()
            print(f"  Iteration time: {iter_end_time - iter_start_time:.2f} seconds")

    else:  # Loop finished without break
        print(f"Warning: Did not converge after {max_iter} iterations.")
        # Return the last successful results if the loop finished without converging
        # This might be partially converged results.
        end_time = time.time()
        if verbose:
             print(f"Total self-consistent loop time: {end_time - start_time:.2f} seconds (Non-converged).")
        return last_successful_results


    # If converged, ensure the returned potential and density are from the final iteration
    # The last_successful_results were updated inside the loop, so they hold the final state upon break.
    end_time = time.time()
    if verbose:
        print(f"Total self-consistent loop time: {end_time - start_time:.2f} seconds (Converged).")

    # Return the final converged state
    return last_successful_results

# Note: The 1D self-consistent solver from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D module if needed.
