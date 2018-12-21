# Density as defined in:
# Hidalgo, C. A., Klinger, B., Barabasi, A.-L., & Hausmann, R. (2007). The Product Space Conditions the Development of Nations. Science, 317(5837), 482–487. https://doi.org/10.1126/science.1144581

import pandas as pd
import numpy as np

def calc_density(rca_or_mcp, proximity_mat):
    """Calculate density, as defined by Hidalgo et. al. (2007)

    Args:
        rca_or_mcp: numpy array of RCA (if continuous product proximities are
            used), else Mcp
        proximity_mat: product proximity matrix

    Returns:
        numpy array of same shape as proximity_mat corresponding to density of
        each product
    """
    den = np.nansum(proximity_mat, axis=1)[np.newaxis, :]
    # density = rca_or_mcp @ (proximity_mat / den)
    density = rca_or_mcp @ (proximity_mat.T / den)
    return(density)
