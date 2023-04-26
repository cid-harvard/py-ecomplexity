import numpy as np
import pandas as pd
import warnings
from ecomplexity.calc_proximity import calc_discrete_proximity
from ecomplexity.calc_proximity import calc_continuous_proximity
from ecomplexity.ComplexityData import ComplexityData
from ecomplexity.calc_density import calc_density
from ecomplexity.coicog import calc_coi_cog


def reshape_output_to_data(cdata, t):
    """Reshape output ndarrays to df"""
    diversity = (
        cdata.diversity_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()
    )
    ubiquity = (
        cdata.ubiquity_t[np.newaxis, :].repeat(cdata.mcp_t.shape[0], axis=0).ravel()
    )
    eci = cdata.eci_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()
    pci = cdata.pci_t[np.newaxis, :].repeat(cdata.mcp_t.shape[0], axis=0).ravel()
    coi = cdata.coi_t[:, np.newaxis].repeat(cdata.mcp_t.shape[1], axis=1).ravel()

    out_dict = {
        "diversity": diversity,
        "ubiquity": ubiquity,
        "mcp": cdata.mcp_t.ravel(),
        "eci": eci,
        "pci": pci,
        "density": cdata.density_t.ravel(),
        "coi": coi,
        "cog": cdata.cog_t.ravel(),
    }

    if hasattr(cdata, "rpop_t"):
        out_dict["rca"] = cdata.rca_t.ravel()
        out_dict["rpop"] = cdata.rpop_t.ravel()

    elif hasattr(cdata, "rca_t"):
        out_dict["rca"] = cdata.rca_t.ravel()

    output = pd.DataFrame.from_dict(out_dict).reset_index(drop=True)

    cdata.data_t["time"] = t
    cdata.output_t = pd.concat([cdata.data_t.reset_index(), output], axis=1)
    cdata.output_list.append(cdata.output_t)
    return cdata


def conform_to_original_data(cdata, data):
    """Reset column names and add dropped columns back"""
    cdata.output = cdata.output.rename(columns=cdata.cols_input)
    cdata.output = cdata.output.merge(
        data, how="outer", on=list(cdata.cols_input.values())
    )
    return cdata


def calc_eci_pci(cdata):
    # Check if diversity or ubiquity is 0 or nan, can cause problems
    if ((cdata.diversity_t == 0).sum() > 0) | ((cdata.ubiquity_t == 0).sum() > 0):
        warnings.warn(
            f"In year {cdata.t}, diversity / ubiquity is 0 for some locs/prods"
        )

    # Extract valid elements only
    cntry_mask = np.argwhere(cdata.diversity_t == 0).squeeze()
    prod_mask = np.argwhere(cdata.ubiquity_t == 0).squeeze()
    diversity_valid = cdata.diversity_t[cdata.diversity_t != 0]
    ubiquity_valid = cdata.ubiquity_t[cdata.ubiquity_t != 0]
    mcp_valid = cdata.mcp_t[cdata.diversity_t != 0, :][:, cdata.ubiquity_t != 0]

    # Calculate ECI and PCI eigenvectors
    mcp1 = mcp_valid / diversity_valid[:, np.newaxis]
    mcp2 = mcp_valid / ubiquity_valid[np.newaxis, :]
    # Make copy of transpose to ensure contiguous array for performance reasons
    mcp2_t = mcp2.T.copy()
    # These matrix multiplication lines are very slow
    Mcc = mcp1 @ mcp2_t
    Mpp = mcp2_t @ mcp1

    try:
        # Calculate eigenvectors
        eigvals, eigvecs = np.linalg.eig(Mpp)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort()[-2]
        kp = eigvecs[:, eig_index]
        kc = mcp1 @ kp

        # Adjust sign of ECI and PCI so it makes sense, as per book
        s1 = np.sign(np.corrcoef(diversity_valid, kc)[0, 1])
        eci_t = s1 * kc
        pci_t = s1 * kp

        # Add back the deleted elements
        for x in cntry_mask:
            eci_t = np.insert(eci_t, x, np.nan)
        for x in prod_mask:
            pci_t = np.insert(pci_t, x, np.nan)

    except Exception as e:
        warnings.warn(f"Unable to calculate eigenvectors for year {cdata.t}")
        print(e)
        eci_t = np.empty(cdata.mcp_t.shape[0])
        pci_t = np.empty(cdata.mcp_t.shape[1])
        eci_t[:] = np.nan
        pci_t[:] = np.nan

    return (eci_t, pci_t)


def ecomplexity(
    data,
    cols_input,
    presence_test="rca",
    val_errors_flag="coerce",
    rca_mcp_threshold=1,
    rpop_mcp_threshold=1,
    pop=None,
    continuous=False,
    proximity_edgelist=None,
    asymmetric=False,
    knn=None,
    verbose=True,
):
    """Complexity calculations through the ComplexityData class

    Args:
        data: pandas dataframe containing production / trade data.
            Including variables indicating time, location, product and value
        cols_input: dict of column names for time, location, product and value.
            Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
        presence_test: str for test used for presence of industry in location.
            One of "rca" (default), "rpop", or "manual".
            Determines which values are used for M_cp calculations.
            If "manual", M_cp is taken as given from the "value" column in data
        val_errors_flag: {'coerce','ignore','raise'}. Passed to pd.to_numeric
            *default* coerce.
        rca_mcp_threshold: numeric indicating RCA threshold beyond which mcp is 1.
            *default* 1.
        rpop_mcp_threshold: numeric indicating RPOP threshold beyond which mcp is 1.
            *default* 1. Only used if presence_test is not "rca".
        pop: pandas df, with time, location and corresponding population, in that order.
            Not required if presence_test is "rca", which is the default.
        continuous: Used to calculate product proximities, indicates whether
            to consider correlation of every product pair (True) or product
            co-occurrence (False). *default* False.
        asymmetric: Used to calculate product proximities, indicates whether
            to generate asymmetric proximity matrix (True) or symmetric (False).
            *default* False.
        proximity_edgelist: pandas df with cols 'prod1', 'prod2', 'proximity'.
            If None (default), proximity values are calculated from data.
        knn: Number of nearest neighbors from proximity matrix to use to calculate
            density. Will use entire proximity matrix if None.
            *default* None.
        verbose: Print year being processed

    Returns:
        Pandas dataframe containing the data with the following additional columns:
            - diversity: k_c,0
            - ubiquity: k_p,0
            - rca: Balassa's RCA
            - rpop: (available if presence_test!="rca") RPOP
            - mcp: MCP used for complexity calculations
            - eci: Economic complexity index
            - pci: Product complexity index
            - density: Density of the network around each product
            - coi: Complexity Outlook Index
            - cog: Complexity Outlook Gain

    """
    cdata = ComplexityData(data, cols_input, val_errors_flag)

    cdata.output_list = []

    # Iterate over time stamps
    for t in cdata.data.index.unique("time"):
        if verbose:
            print(t)
        # Rectangularize df
        cdata.create_full_df(t)

        # Check if Mcp is pre-computed
        if presence_test != "manual":
            cdata.calculate_rca()
            cdata.calculate_mcp(
                rca_mcp_threshold, rpop_mcp_threshold, presence_test, pop, t
            )
        else:
            cdata.calculate_manual_mcp()

        # Calculate diversity and ubiquity
        cdata.diversity_t = np.nansum(cdata.mcp_t, axis=1)
        cdata.ubiquity_t = np.nansum(cdata.mcp_t, axis=0)

        # If ANY of diversity or ubiquity is 0, warn that eci and pci will be nan
        if np.any(cdata.diversity_t == 0) or np.any(cdata.ubiquity_t == 0):
            warnings.warn(
                f"Year {t}: Diversity or ubiquity is 0, so ECI and PCI will be nan"
            )

        # Calculate ECI and PCI
        cdata.eci_t, cdata.pci_t = calc_eci_pci(cdata)

        # Check if proximities are pre-computed, otherwise compute from data
        if proximity_edgelist is not None:
            # Take proximity edgelist and convert to matrix
            prox_mat = proximity_edgelist.pivot(
                index="prod1", columns="prod2", values="proximity"
            )
            # Make sure that the set of products in prod1 and prod2 are the same
            # and make sure it is a square matrix
            assert set(list(proximity_edgelist["prod1"].unique())) == set(
                list(proximity_edgelist["prod2"].unique())
            ), "The set of products in prod1 and prod2 are not the same"
            assert prox_mat.shape[0] == prox_mat.shape[1], "prox_mat is not square"

            # Reindex
            prox_mat = prox_mat.reindex(cdata.data_t.index.levels[1])
            prox_mat = prox_mat.reindex(cdata.data_t.index.levels[1], axis=1)
            # Get values
            prox_mat = prox_mat.values
            # Check if any values are nan. If any nan's are present, warn
            if np.any(np.isnan(prox_mat)):
                # Get fraction of values that are nan that are not diagonal elements
                prox_mat_nan_check = prox_mat.copy()
                np.fill_diagonal(prox_mat_nan_check, 1)
                nan_frac = (
                    np.sum(np.isnan(prox_mat_nan_check)) / prox_mat_nan_check.size
                )
                if nan_frac > 0:
                    warnings.warn(
                        f"Year {t}: Proximity matrix contains {nan_frac*100:.2}% non-diagonal values that are NaN's, so some density values will be NaN.\nAssuming diagonals are 1 and that other nan's are zero."
                    )
                else:
                    warnings.warn(
                        f"Year {t}: Proximity matrix contains diagonal values that are NaN's. Assuming all diagonal values to be one."
                    )
                # Replace diagonals with one
                np.fill_diagonal(prox_mat, 1)
                # Replace other nan's with zero
                prox_mat[np.isnan(prox_mat)] = 0

        else:
            # Calculate proximity
            if continuous == False:
                prox_mat = calc_discrete_proximity(
                    cdata.mcp_t, cdata.ubiquity_t, asymmetric
                )
            elif continuous == True and presence_test == "rpop":
                prox_mat = calc_continuous_proximity(cdata.rpop_t, cdata.ubiquity_t)
            elif continuous == True and presence_test != "rpop":
                prox_mat = calc_continuous_proximity(cdata.rca_t, cdata.ubiquity_t)

        # Calculate density
        # If there are any nulls in the proximity matrix, drop
        if continuous == False or presence_test == "manual":
            cdata.density_t = calc_density(
                rca_or_mcp=cdata.mcp_t, proximity_mat=prox_mat, knn=knn
            )
        elif continuous == True and presence_test == "rpop":
            cdata.density_t = calc_density(
                rca_or_mcp=cdata.rpop_t, proximity_mat=prox_mat, knn=knn
            )
        elif continuous == True and presence_test == "rca":
            cdata.density_t = calc_density(
                rca_or_mcp=cdata.rca_t, proximity_mat=prox_mat, knn=knn
            )

        # Calculate COI and COG
        cdata.coi_t, cdata.cog_t = calc_coi_cog(cdata, prox_mat)

        # Normalize variables as per STATA package
        # Normalization using ECI mean and std. dev. preserves the property that
        # ECI = (mean of PCI of products for which MCP=1)
        cdata.pci_t = (cdata.pci_t - cdata.eci_t.mean()) / cdata.eci_t.std()
        cdata.cog_t = cdata.cog_t / cdata.eci_t.std()
        cdata.eci_t = (cdata.eci_t - cdata.eci_t.mean()) / cdata.eci_t.std()

        cdata.coi_t = (cdata.coi_t - cdata.coi_t.mean()) / cdata.coi_t.std()

        # Reshape ndarrays to df
        cdata = reshape_output_to_data(cdata, t)

    cdata.output = pd.concat(cdata.output_list)
    cdata = conform_to_original_data(cdata, data)

    return cdata.output
