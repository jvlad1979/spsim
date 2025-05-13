
import numpy as np
from ..constants import e # Import elementary charge

def calculate_charge_density_2d(eigenvalues, eigenvectors_2d, fermi_level):
    """
    Calculates the 2D electron charge density (C/m^2).
    Assumes zero temperature.

    Args:
        eigenvalues (np.ndarray): Array of energy eigenvalues in Joules.
        eigenvectors_2d (np.ndarray): 3D array (Nx, Ny, num_eigenstates) of wavefunctions.
        fermi_level (float): Fermi level in Joules.

    Returns:
        np.ndarray: 2D array of electron charge density in C/m^2.
    """
    # Note: This implementation assumes zero temperature (Fermi-Dirac is a step function).
    # For finite temperature, the Fermi-Dirac distribution should be used:
    # temperature = 1.0 # Kelvin
    # kT = const.k * temperature
    # fermi_dirac = 1 / (1 + np.exp((eigenvalues[i] - fermi_level) / kT))

    Nx, Ny, num_states = eigenvectors_2d.shape
    density_2d = np.zeros((Nx, Ny))  # electrons/m^2

    # Factor of 2 for spin degeneracy
    for i in range(num_states):
        if eigenvalues[i] < fermi_level:
            density_2d += 2 * np.abs(eigenvectors_2d[:, :, i]) ** 2
        else:
            # Assuming eigenvalues are sorted, we can break early
            break

    # Charge density (negative for electrons)
    charge_density_2d = -e * density_2d  # Coulombs per square meter (C/m^2)
    return charge_density_2d

def calculate_total_electrons(charge_density_2d, dx, dy):
    """
    Calculates the total number of electrons by integrating the charge density.

    Args:
        charge_density_2d (np.ndarray): 2D array of charge density in C/m^2.
        dx (float): Grid spacing in x (m).
        dy (float): Grid spacing in y (m).

    Returns:
        float: Total number of electrons.
    """
    # Ensure charge_density_2d is a numpy array
    charge_density_2d = np.asarray(charge_density_2d)

    # Integrate density (electrons/m^2) over area (dx*dy)
    # charge_density_2d is in C/m^2, divide by -e to get electrons/m^2
    electron_density_per_m2 = charge_density_2d / (-e)
    total_electrons = np.sum(electron_density_per_m2 * dx * dy)
    return total_electrons

# Note: The 1D charge density calculation from simulate_1d_dot.py
# is not included here as it's 1D specific. It could be added
# to a separate 1D module if needed.
