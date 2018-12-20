import pandas as pd
import numpy as np

def proximity(data, rca_or_mcp, ubiquity, continuous=False, asymmetric=False):
    """Wrapper function to calculate product proximity matrices

    Args:
        data: pandas df with cols 'time','loc','prod','val'
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
    # Reshape as df
    output_index = pd.MultiIndex.from_product([data.index.levels[1], data.index.levels[1]])
    output = pd.DataFrame(data={'proximity':phi.ravel()},
                          index=output_index)
    output.head()
