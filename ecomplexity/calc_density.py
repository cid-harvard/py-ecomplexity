# Density as defined in:
# Hidalgo, C. A., Klinger, B., Barabasi, A.-L., & Hausmann, R. (2007). The Product Space Conditions the Development of Nations. Science, 317(5837), 482–487. https://doi.org/10.1126/science.1144581

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


def calc_density(rca_or_mcp, proximity_mat, knn=None):
    """Calculate density, as defined by Hidalgo et. al. (2007)

    Args:
        rca_or_mcp: numpy array of RCA (if continuous product proximities are
            used), else Mcp
        proximity_mat: product proximity matrix
        knn: number of nearest neighbors to consider for density calculation (optional)

    Returns:
        numpy array of same shape as proximity_mat corresponding to density of
        each product
    """
    if knn is None:
        den = np.nansum(proximity_mat, axis=1)[np.newaxis, :]
        # density = rca_or_mcp @ (proximity_mat / den)
        density = rca_or_mcp @ (proximity_mat.T / den)
    else:
        # Convert proximity matrix to a distance matrix
        distance_mat = 1 - proximity_mat
        # Get proximity to k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=knn, metric="precomputed").fit(distance_mat)
        distance_knn, indices_knn = nbrs.kneighbors()
        # Get proximity
        proximity_knn = 1 - distance_knn
        # Calculate density
        # Get denominator
        den = np.nansum(proximity_knn, axis=1)
        density = []
        for i, row in enumerate(indices_knn):
            # Use row to subset rca_or_mcp
            rca_knn_p = rca_or_mcp[np.arange(rca_or_mcp.shape[0])[:, np.newaxis], row]
            # Get distance_knn for this row
            proximity_knn_row = proximity_knn[i]
            # Divide by den
            proximity_knn_row = proximity_knn_row / den[i]
            # Multiply each row of rca_knn_p by proximity_knn_row
            num_p = rca_knn_p * proximity_knn_row
            # Sum across columns
            density_p = np.nansum(rca_knn_p, axis=1)
            density.append(density_p)
        density = np.array(density).T
    return density
