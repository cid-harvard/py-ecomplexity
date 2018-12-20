import pandas as pd
import numpy as np

def calc_discrete_proximity(mcp, ubiquity, asymmetric=False):
    """Calculate product proximity matrices - discrete

    Args:
        mcp: numpy ndarray with rows as locations and columns as products
        ubiquity: numpy array of shape=number of columns in "rca_or_mcp"
        asymmetric: Whether to generate asymmetric proximity matrix (True) or
            symmetric (False). *default* False.

    Returns:
        pandas df with proximity values for every product pair
    """

    # Calculate discrete proximity
    phi = mcp.T @ mcp
    phi = (phi / ubiquity[np.newaxis, :])

    if asymmetric==False:
        # Symmetric proximity matrix
        phi = np.minimum(phi, phi.T)
    elif asymmetric==True:
        # Asymmetric proximity matrix
        phi = phi.T

    return(phi)

def calc_continuous_proximity(rca, ubiquity):
    """Calculate product proximity matrices - continuous

    Args:
        rca: numpy ndarray with rows as locations and columns as products
        ubiquity: numpy array of shape=number of columns in "rca_or_mcp"

    Returns:
        pandas df with proximity values for every product pair
    """
    # Calculate continuous proximity
    phi = (1 + np.corrcoef(rca.T))/2
    return(phi)
