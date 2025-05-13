
import numpy as np
import time
from ..constants import e # Import elementary charge
from ..device.potential import get_external_potential # Import external potential function
from ..solvers.schrodinger import solve_schrodinger_2d # Import Schrödinger solver
from ..solvers.poisson import solve_poisson_2d_fd, solve_poisson_2d_spectral, solve_poisson_2d_fem_stub # Import Poisson solvers
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

    # Initialize variables to store the final results
    final_charge_density = None
    # Initialize final_electrostatic_potential_V with the initial guess or zero potential
    final_electrostatic_potential_V = electrostatic_potential_V.copy()


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
            # Return None for charge density and the current potential on failure
            return None, electrostatic_potential_V

        if verbose:
            if eigenvalues.size > 0:
                print(f"  Min Eigenvalue: {eigenvalues[0]/e:.4f} eV, Fermi Level: {fermi_level/e:.4f} eV")
            else:
                print("  No eigenvalues found.")

        # 3. Calculate charge density
        new_charge_density = calculate_charge_density_2d(
            eigenvalues, eigenvectors_2d, fermi_level
        )
        # Update final charge density candidate
        final_charge_density = new_charge_density

        # 4. Solve Poisson equation
        # Pass grid info to the solver
        new_electrostatic_potential_V = poisson_solver_func(new_charge_density, Nx, Ny, dx, dy)

        # 5. Check for convergence (using norm of potential difference)
        potential_diff_norm = np.linalg.norm(
            new_electrostatic_potential_V - electrostatic_potential_V
        ) * np.sqrt(dx * dy) # Use sqrt(dx*dy) for 2D norm scaling

        if verbose:
            print(f"  Potential difference norm: {potential_diff_norm:.3e}")

        # Update final potential candidate (potential before mixing)
        final_electrostatic_potential_V = new_electrostatic_potential_V

        if potential_diff_norm < tol:
            if verbose:
                print(f"Converged after {i + 1} iterations.")
            # If converged, the final potential is new_electrostatic_potential_V
            final_electrostatic_potential_V = new_electrostatic_potential_V
            break

        # 6. Mix potential for stability
        # Simple linear mixing
        electrostatic_potential_V = electrostatic_potential_V + mixing * (
            new_electrostatic_potential_V - electrostatic_potential_V
        )
        # After mixing, electrostatic_potential_V is the potential for the next iteration.
        # If the loop finishes without converging, this will be the final potential.


        if verbose:
            iter_end_time = time.time()
            print(f"  Iteration time: {iter_end_time - iter_start_time:.2f} seconds")

    else:  # Loop finished without break
        print(f"Warning: Did not converge after {max_iter} iterations.")
        end_time = time.time()
        if verbose:
             print(f"Total self-consistent loop time: {end_time - start_time:.2f} seconds (Non-converged).")
        # If not converged, the final potential is the result of the last mixing step.
        final_electrostatic_potential_V = electrostatic_potential_V


    # Return the final charge density and the final electrostatic potential
    # If Schrödinger solver failed, final_charge_density is None.
    end_time = time.time()
    if verbose:
        print(f"Total self-consistent loop time: {end_time - start_time:.2f} seconds.")

    return final_charge_density, final_electrostatic_potential_V

# Note: The 1D self-consistent solver from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D module if needed.
