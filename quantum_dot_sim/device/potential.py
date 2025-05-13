
import numpy as np
from ..constants import e # Import elementary charge from constants

def get_external_potential(X, Y, voltages, Lx, Ly):
    """
    Calculates the 2D external potential profile based on gate voltages.
    Uses 2D Gaussian profiles for gate influence.

    Args:
        X (np.ndarray): 2D meshgrid of x-coordinates.
        Y (np.ndarray): 2D meshgrid of y-coordinates.
        voltages (dict): Dictionary mapping gate names (str) to applied voltages (float).
        Lx (float): Length of the simulation domain in x (m).
        Ly (float): Length of the simulation domain in y (m).

    Returns:
        np.ndarray: 2D array of the external potential profile in Joules.
    """
    potential = np.zeros_like(X)  # Potential in Joules

    # Define gate parameters (centers and widths in meters)
    # Example: A double dot defined by top gates
    # These values are hardcoded here but could be moved to a config or function args
    gate_std_dev_x = 20e-9
    gate_std_dev_y = 20e-9  # Can be different for x and y

    # Gate positions (example) - Same as simulate_2d_dot
    p1_center = (Lx * 0.35, Ly * 0.5)
    p2_center = (Lx * 0.65, Ly * 0.5)
    b1_center = (Lx * 0.15, Ly * 0.5)  # Left barrier
    b2_center = (Lx * 0.50, Ly * 0.5)  # Center barrier
    b3_center = (Lx * 0.85, Ly * 0.5)  # Right barrier

    # Helper function for 2D Gaussian potential shape
    def gaussian_potential_2d(center_x, center_y, std_dev_x, std_dev_y, amplitude):
        return amplitude * np.exp(
            -(
                (X - center_x) ** 2 / (2 * std_dev_x**2)
                + (Y - center_y) ** 2 / (2 * std_dev_y**2)
            )
        )

    # Add potential contributions from each gate
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

    # Optional: Add confinement in y-direction (e.g., parabolic)
    # potential += 0.5 * m_eff * (omega_y**2) * (Y - Ly/2)**2

    return potential  # Potential in Joules

# Note: The 1D external potential function from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D device module if needed.
