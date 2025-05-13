
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve # Assuming this library is installed

def get_hilbert_order(nx, ny):
    """
    Generates a list of (i, j) indices in Hilbert curve order for an nx x ny grid.

    Args:
        nx (int): Number of points in the x-dimension of the grid.
        ny (int): Number of points in the y-dimension of the grid.

    Returns:
        list: A list of (i, j) tuples representing grid indices in Hilbert order.
    """
    # Determine the Hilbert curve level p such that 2^p >= max(nx, ny)
    # The hilbertcurve library requires the grid dimensions to be powers of 2
    # for a perfect curve, but it can map points within a 2^p x 2^p space.
    # We need p such that 2^p is large enough to contain the nx x ny grid.
    p = int(np.ceil(np.log2(max(nx, ny))))
    n_dims = 2
    hilbert_curve = HilbertCurve(p, n_dims)

    # Generate all grid points (indices) within the nx x ny bounds
    points = [(i, j) for i in range(nx) for j in range(ny)]

    # Calculate Hilbert distances for each point
    # The points_to_distances method expects a list of lists/tuples
    distances = hilbert_curve.distances_from_points(points)

    # Sort points based on Hilbert distances
    hilbert_ordered_indices = [point for _, point in sorted(zip(distances, points))]

    return hilbert_ordered_indices
