import numpy as np
import pandas as pd
from ecomplexity.calc_proximity import calc_discrete_proximity
from ecomplexity.calc_proximity import calc_continuous_proximity
from ecomplexity.ComplexityData import ComplexityData
from ecomplexity.density import calc_density

def reshape_output_to_data(cdata, t):
    """Reshape output ndarrays to df"""
    diversity = cdata.diversity_t[:, np.newaxis].repeat(
        cdata.mcp_t.shape[1], axis=1).ravel()
    ubiquity = cdata.ubiquity_t[np.newaxis, :].repeat(
        cdata.mcp_t.shape[0], axis=0).ravel()
    eci = cdata.eci_t[:, np.newaxis].repeat(
        cdata.mcp_t.shape[1], axis=1).ravel()
    pci = cdata.pci_t[np.newaxis, :].repeat(
        cdata.mcp_t.shape[0], axis=0).ravel()

    out_dict = {'diversity': diversity,
                'ubiquity': ubiquity,
                'mcp': cdata.mcp_t.ravel(),
                'eci': eci,
                'pci': pci,
                'density': cdata.density_t.ravel()}

    if hasattr(cdata, 'rpop_t'):
        out_dict['rca'] = cdata.rca_t.ravel()
        out_dict['rpop'] = cdata.rpop_t.ravel()

    elif hasattr(cdata, 'rca_t'):
        out_dict['rca'] = cdata.rca_t.ravel()

    output = pd.DataFrame.from_dict(out_dict).reset_index(drop=True)

    cdata.data_t['time'] = t
    cdata.output_t = pd.concat([cdata.data_t.reset_index(), output], axis=1)
    cdata.output_list.append(cdata.output_t)
    return(cdata)


def conform_to_original_data(cdata, cols_input, data):
    """Reset column names and add dropped columns back"""
    cdata.output = cdata.output.rename(columns=cols_input)
    cdata.output = cdata.output.merge(
        data, how="outer", on=list(cols_input.values()))
    return(cdata)


def calculate_Kvec(cdata):
    """Calculate eigenvectors for PCI and ECI"""
    mcp1 = (cdata.mcp_t / cdata.diversity_t[:, np.newaxis])
    mcp2 = (cdata.mcp_t / cdata.ubiquity_t[np.newaxis, :])
    # These matrix multiplication lines are very slow
    Mcc = mcp1 @ mcp2.T
    Mpp = mcp2.T @ mcp1

    # Calculate eigenvectors
    eigvals, eigvecs = np.linalg.eig(Mpp)
    eigvecs = np.real(eigvecs)
    # Get eigenvector corresponding to second largest eigenvalue
    eig_index = eigvals.argsort()[-2]
    kp = eigvecs[:, eig_index]
    kc = mcp1 @ kp
    return(kp, kc)


def sign(diversity, eci):
    return(np.sign(np.corrcoef(diversity, eci)[0, 1]))

def calc_eci_pci(cdata):
    # Calculate ECI and PCI eigenvectors
    kp, kc = calculate_Kvec(cdata)

    # Adjust sign of ECI and PCI so it makes sense, as per book
    s1 = sign(cdata.diversity_t, kc)
    cdata.eci_t = s1 * kc
    cdata.pci_t = s1 * kp

    # Normalize ECI and PCI (based on ECI values)
    cdata.pci_t = (cdata.pci_t - cdata.eci_t.mean()) / cdata.eci_t.std()
    cdata.eci_t = (cdata.eci_t - cdata.eci_t.mean()) / cdata.eci_t.std()
    return(cdata)


def ecomplexity(data, cols_input, presence_test="rca", val_errors_flag='coerce',
                rca_mcp_threshold=1, rpop_mcp_threshold=1, pop=None,
                continuous=False, asymmetric=False):
    """Complexity calculations through the ComplexityData class

    Args:
        data: pandas dataframe containing production / trade data.
            Including variables indicating time, location, product and value
        cols_input: dict of column names for time, location, product and value.
            Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
        presence_test: str for test used for presence of industry in location.
            One of "rca" (default), "rpop", "both", or "manual".
            Determines which values are used for M_cp calculations.
            If "manual", M_cp is taken as given from the "value" column in data
        val_errors_flag: {'coerce','ignore','raise'}. Passed to pd.to_numeric
            *default* coerce.
        rca_mcp_threshold: numeric indicating RCA threshold beyond which mcp is 1.
            *default* 1.
        rpop_mcp_threshold: numeric indicating RPOP threshold beyond which mcp is 1.
            *default* 1. Only used if presence_test is not "rca".
        pop: pandas df, with time, location and corresponding population, in that order.
            Not required if presence_test is "rca" (default).
        continuous: Used to calculate product proximities, indicates whether
            to consider correlation of every product pair (True) or product
            co-occurrence (False). *default* False.
        asymmetric: Used to calculate product proximities, indicates whether
            to generate asymmetric proximity matrix (True) or symmetric (False).
            *default* False.

    Returns:
        Pandas dataframe containing the data with the following additional columns:
            - diversity: k_c,0
            - ubiquity: k_p,0
            - rca: Balassa's RCA
            - rpop: (available if presence_test!="rca") RPOP
            - mcp: MCP used for complexity calculations
            - eci: Economic complexity index
            - pci: Product complexity index

    """
    cdata = ComplexityData(data, cols_input, val_errors_flag)

    cdata.output_list = []

    # Iterate over time stamps
    for t in cdata.data.index.unique("time"):
        print(t)
        # Rectangularize df
        cdata.create_full_df(t)

        # Check if Mcp is pre-computed
        if presence_test != "manual":
            cdata.calculate_rca()
            cdata.calculate_mcp(rca_mcp_threshold, rpop_mcp_threshold,
                                presence_test, pop, t)
        else:
            cdata.calculate_manual_mcp()

        # Calculate diversity and ubiquity
        cdata.diversity_t = np.nansum(cdata.mcp_t, axis=1)
        cdata.ubiquity_t = np.nansum(cdata.mcp_t, axis=0)

        # Calculate ECI and PCI
        cdata = calc_eci_pci(cdata)

        # Calculate proximity and density
        if continuous == False:
            prox_mat = calc_discrete_proximity(cdata.mcp_t, cdata.ubiquity_t,
                                               asymmetric)
            cdata.density_t = calc_density(cdata.mcp_t, prox_mat)
        elif continuous == True and presence_test == "rpop":
            prox_mat = calc_continuous_proximity(
                cdata.rpop_t, cdata.ubiquity_t)
            cdata.density_t = calc_density(cdata.rpop_t, prox_mat)
        elif continuous == True and presence_test != "rpop":
            prox_mat = calc_continuous_proximity(cdata.rca_t, cdata.ubiquity_t)
            cdata.density_t = calc_density(cdata.rca_t, prox_mat)

        # # Calculate COI and COG
        # cdata = calc_coi_cog(cdata)

        # Reshape ndarrays to df
        cdata = reshape_output_to_data(cdata, t)

    cdata.output = pd.concat(cdata.output_list)
    cdata = conform_to_original_data(cdata, cols_input, data)

    return(cdata.output)
