import pandas as pd
import numpy as np

def calc_proximity(rca_or_mcp, ubiquity, continuous=False, asymmetric=False):
    """Calculate product proximity matrices

    Args:
        rca_or_mcp: numpy ndarray with rows as locations and columns as products.
            Considered to be RCA if continuous is True. Else considered as Mcp.
        ubiquity: numpy array of shape=number of columns in "rca_or_mcp"
        continuous: Whether to consider correlation of every product pair (True)
            or product co-occurrence (False). *default* False.
        asymmetric: Whether to generate asymmetric proximity matrix (True) or
            symmetric (False). *default* False.

    Returns:
        pandas df with proximity values for every product pair
    """
    if continuous==False:
        # Calculate discrete proximity
        mcp = rca_or_mcp
        phi = mcp.T @ mcp
        phi = (phi / ubiquity[np.newaxis, :])

        if asymmetric==False:
            # Symmetric proximity matrix
            phi = np.minimum(phi, phi.T) - np.identity(mcp.shape[1])
        elif asymmetric==True:
            # Asymmetric proximity matrix
            phi = phi.T - np.identity(mcp.shape[1])

    elif continuous==True:
        # Calculate continuous proximity
        rca = rca_or_mcp
        phi = (1 + np.corrcoef(rca.T))/2 - np.identity(rca.shape[1])

    return(phi)
