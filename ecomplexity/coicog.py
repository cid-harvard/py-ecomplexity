import pandas as pd
import numpy as np

def calc_coi_cog(cdata, proximity_mat):
    """Calculate Complexity Outlook index

    Args:
        cdata: Object of ComplexityData class, with density calculated
        proximity_mat: proximity matrix

    Returns:
        cata: ComplexityData object with attribute coi
    """
    # mata coi = ((density:*(1 :- M)):*kp)*J(Npx,Npx,1)
    coi = ((cdata.density_t * (1-cdata.mcp_t)) * cdata.pci_t).sum(axis=1)
    print(coi.shape)
    # mata cog = (1 :- M):*((1 :- M) * (proximity :* ((kp1d:/(proximity*J(Npx,1,1)))*J(1,Npx,1))))
    cog = (1-cdata.mcp_t) * ((1-cdata.mcp_t) @ (proximity_mat * (cdata.pci_t / proximity_mat.sum(axis=1))[:, np.newaxis]))
    print(cog.shape)
    return(coi, cog)
