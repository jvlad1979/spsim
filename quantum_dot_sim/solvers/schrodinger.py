
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..constants import hbar, m_eff # Import constants

def solve_schrodinger_2d(potential_2d, Nx, Ny, dx, dy, num_eigenstates=10, solver_config=None):
    """
    Solves the 2D time-independent Schrödinger equation.
    Returns eigenvalues (energies) and eigenvectors (wavefunctions reshaped to 2D).

    Args:
        potential_2d (np.ndarray): 2D array of the potential energy in Joules.
        Nx (int): Number of grid points in x.
        Ny (int): Number of grid points in y.
        dx (float): Grid spacing in x (m).
        dy (float): Grid spacing in y (m).
        num_eigenstates (int): Number of lowest eigenstates to find.
        solver_config (dict, optional): Configuration for the solver (e.g., {'method': 'eigsh', 'params': {'which': 'SM'}}).
                                         Defaults to eigsh with 'SM'.

    Returns:
        tuple: (eigenvalues, eigenvectors_2d)
               eigenvalues (np.ndarray): Array of energy eigenvalues in Joules.
               eigenvectors_2d (np.ndarray): 3D array (Nx, Ny, num_eigenstates) of wavefunctions.
               Returns (np.array([]), np.empty((Nx, Ny, 0))) if solver fails.
    """
    N_total = Nx * Ny
    potential_flat = potential_2d.flatten(order="C")  # Flatten consistently

    # Construct the 2D Hamiltonian using sparse matrix format
    # H = -hbar^2/(2*m_eff) * (d^2/dx^2 + d^2/dy^2) + V(x,y)
    # Finite difference approximation (5-point stencil)

    # Diagonal terms
    diag = hbar**2 / (m_eff * dx**2) + hbar**2 / (m_eff * dy**2) + potential_flat
    # Off-diagonal terms for x-derivatives
    offdiag_x = -(hbar**2) / (2 * m_eff * dx**2) * np.ones(N_total)
    # Off-diagonal terms for y-derivatives
    offdiag_y = -(hbar**2) / (2 * m_eff * dy**2) * np.ones(N_total)

    # Create sparse matrix diagonals
    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    # Define offsets for diagonals
    # 0: main diagonal
    # -1, 1: x-neighbors (careful with boundaries)
    # -Nx, Nx: y-neighbors
    offsets = [0, -1, 1, -Nx, Nx]

    # Adjust off-diagonals at boundaries (where stencil wraps around)
    # Remove connections between y=0 and y=Ny-1 (for offset -1)
    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    # Remove connections between y=Ny-1 and y=0 (for offset 1)
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0  # Corrected index

    H = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    # Default solver config
    if solver_config is None:
        solver_config = {"method": "eigsh", "params": {"which": "SM"}}

    method = solver_config.get("method", "eigsh")
    params = solver_config.get("params", {})

    eigenvalues = np.array([])
    eigenvectors_flat = np.empty((N_total, 0))

    try:
        if method == "eigsh":
            eigsh_params = params.copy()
            # Ensure k is passed from function argument if not in params
            if 'k' not in eigsh_params:
                 eigsh_params['k'] = num_eigenstates

            # Handle sigma option if present
            if eigsh_params.pop("use_sigma_min_potential", False):
                sigma = np.min(potential_flat)
                eigsh_params["sigma"] = sigma
                if "which" not in eigsh_params:
                    eigsh_params["which"] = "LM" # sigma requires LM or LA

            # Remove non-eigsh specific params
            eigsh_params.pop("use_random_X", None)

            eigenvalues, eigenvectors_flat = spla.eigsh(H, **eigsh_params)

        elif method == "lobpcg":
            lobpcg_params = params.copy()
            k_lobpcg = lobpcg_params.pop("k", num_eigenstates) # Use k from params or default

            if lobpcg_params.pop("use_random_X", False):
                X_init = np.random.rand(N_total, k_lobpcg)
            else:
                 # Default to random if no other X strategy is defined for LOBPCG
                 X_init = np.random.rand(N_total, k_lobpcg)

            # Ensure 'tol' and 'maxiter' are present with defaults if not in params
            if "tol" not in lobpcg_params:
                lobpcg_params["tol"] = 1e-5
            if "maxiter" not in lobpcg_params:
                lobpcg_params["maxiter"] = N_total // 2

            eigenvalues, eigenvectors_flat = spla.lobpcg(
                H, X_init, largest=False, **lobpcg_params
            )
            # Sort eigenvalues and eigenvectors as LOBPCG might not return them sorted
            idx_sort = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors_flat = eigenvectors_flat[:, idx_sort]

        else:
            print(f"Unsupported Schrödinger solver method: {method}. Using eigsh_SM fallback.")
            # Fallback to default eigsh_SM
            eigenvalues, eigenvectors_flat = spla.eigsh(H, k=num_eigenstates, which="SM")


    except Exception as e:
        print(f"Eigenvalue solver {method} failed: {e}")
        return np.array([]), np.empty((Nx, Ny, 0))

    # Normalize and reshape eigenvectors
    actual_computed_states = eigenvectors_flat.shape[1]
    eigenvectors = np.zeros((Nx, Ny, actual_computed_states)) # Use actual computed count
    for i in range(actual_computed_states):
        psi_flat = eigenvectors_flat[:, i]
        norm_sq = np.sum(np.abs(psi_flat) ** 2) * dx * dy
        if norm_sq > 1e-12:  # Avoid division by zero for zero vectors
            norm = np.sqrt(norm_sq)
            eigenvectors[:, :, i] = (psi_flat / norm).reshape((Nx, Ny), order="C")
        else:
            eigenvectors[:, :, i] = psi_flat.reshape(
                (Nx, Ny), order="C"
            )  # Keep as is if norm is zero


    return eigenvalues, eigenvectors  # Eigenvalues in J, eigenvectors are 2D arrays

# Note: The 1D Schrödinger solver from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D solver module if needed.
