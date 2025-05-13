
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.fft as fft
from ..constants import epsilon # Import permittivity

# Global K_sq for spectral solver (needs to be calculated based on grid size)
# This should ideally be handled within the spectral solver function or passed in.
# For now, define a placeholder and note that it needs grid info.
# K_sq = None # Placeholder

def solve_poisson_2d_fd(charge_density_2d, Nx, Ny, dx, dy):
    """
    Solves the 2D Poisson equation: laplacian(phi) = -rho / epsilon
    Returns the 2D electrostatic potential phi (Volts).
    Uses finite differences and assumes Dirichlet boundary conditions (phi=0 on boundary).

    Args:
        charge_density_2d (np.ndarray): 2D array of charge density in C/m^2.
        Nx (int): Number of grid points in x.
        Ny (int): Number of grid points in y.
        dx (float): Grid spacing in x (m).
        dy (float): Grid spacing in y (m).

    Returns:
        np.ndarray: 2D array of the electrostatic potential in Volts.
                    Returns a zero array if solver fails.
    """
    N_total = Nx * Ny
    rho_flat = charge_density_2d.flatten(order="C")

    # Construct the 2D Laplacian matrix (similar to Hamiltonian kinetic part)
    diag = (-2 / dx**2 - 2 / dy**2) * np.ones(N_total)
    offdiag_x = (1 / dx**2) * np.ones(N_total)
    offdiag_y = (1 / dy**2) * np.ones(N_total)

    diagonals = [diag, offdiag_x[:-1], offdiag_x[:-1], offdiag_y[:-Nx], offdiag_y[:-Nx]]
    offsets = [0, -1, 1, -Nx, Nx]

    # Adjust off-diagonals at boundaries
    for i in range(1, Nx):
        diagonals[1][i * Ny - 1] = 0.0
    for i in range(Nx - 1):
        diagonals[2][(i + 1) * Ny - 1] = 0.0

    A = sp.diags(diagonals, offsets, shape=(N_total, N_total), format="csc")

    # Right-hand side vector b = -rho / epsilon
    b = -rho_flat / epsilon

    # Apply Dirichlet boundary conditions (phi=0 on all edges)
    A = A.tolil()
    b_modified = b.copy()

    # Indices for boundary points
    boundary_indices = []
    # x=0 and x=Nx-1 boundaries
    boundary_indices.extend(range(0, N_total, Ny))  # i=0, all j
    boundary_indices.extend(range(Ny - 1, N_total, Ny))  # i=Nx-1, all j
    # y=0 and y=Ny-1 boundaries (excluding corners already added)
    boundary_indices.extend(range(1, Ny - 1))  # j=0, 0<i<Nx-1
    boundary_indices.extend(range(N_total - Ny + 1, N_total - 1))  # j=Ny-1, 0<i<Nx-1

    # Remove duplicates and sort
    boundary_indices = sorted(list(set(boundary_indices)))

    for idx in boundary_indices:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]
        b_modified[idx] = 0.0

    A = A.tocsc()

    # Solve the linear system A * phi = b_modified
    try:
        phi_flat = spla.spsolve(A, b_modified)
    except spla.MatrixRankWarning:
        print(
            "Warning: Poisson matrix is singular or near-singular. Using iterative solver (GMRES)."
        )
        phi_flat, info = spla.gmres(A, b_modified, tol=1e-8, maxiter=2 * N_total)
        if info != 0:
            print(f"Poisson solver (GMRES) did not converge (info={info}).")
            phi_flat = np.zeros_like(rho_flat)  # Fallback
    except Exception as e:
        print(f"Poisson solver failed: {e}")
        phi_flat = np.zeros_like(rho_flat)  # Fallback

    phi_2d = phi_flat.reshape((Nx, Ny), order="C")
    return phi_2d  # Electrostatic potential in Volts


def solve_poisson_2d_spectral(charge_density_2d, Nx, Ny, dx, dy):
    """
    Solves the 2D Poisson equation: laplacian(phi) = -rho / epsilon
    Returns the 2D electrostatic potential phi (Volts).
    Uses spectral methods (FFT) and assumes periodic boundary conditions.

    Args:
        charge_density_2d (np.ndarray): 2D array of charge density in C/m^2.
        Nx (int): Number of grid points in x.
        Ny (int): Number of grid points in y.
        dx (float): Grid spacing in x (m).
        dy (float): Grid spacing in y (m).

    Returns:
        np.ndarray: 2D array of the electrostatic potential in Volts.
    """
    # Define k-space grid and K_sq if not already defined or if grid size changed
    # This calculation should ideally be done once for a given grid size.
    kx = 2 * np.pi * fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * fft.fftfreq(Ny, d=dy)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    K_sq = Kx**2 + Ky**2

    # 1. FFT of charge density
    rho_k = fft.fft2(charge_density_2d)

    # 2. Solve in Fourier space: phi_k = -rho_k / (epsilon * K_sq)
    # Handle the DC component (k=0,0) separately.
    # For periodic BCs, the average potential is arbitrary.
    # Setting phi_k[0,0] = 0 corresponds to zero average potential.
    K_sq_solver = K_sq.copy()
    # Avoid division by zero for the DC term. Set K_sq[0,0] to 1.0 as phi_k[0,0] will be set to 0.
    if K_sq_solver[0, 0] == 0:
         K_sq_solver[0, 0] = 1.0

    phi_k = -rho_k / (epsilon * K_sq_solver)
    phi_k[0, 0] = 0.0 # Explicitly set DC component to zero

    # 3. Inverse FFT to get potential in real space
    phi_2d = fft.ifft2(phi_k).real # Take real part as potential is real

    return phi_2d # Electrostatic potential in Volts


def solve_poisson_2d_fem_stub(charge_density_2d, Nx, Ny, dx, dy):
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

# Note: The 1D Poisson solver from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D solver module if needed.
