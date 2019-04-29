import pandas as pd
import numpy as np
import tqdm

from ecomplexity.calc_proximity import calc_discrete_proximity
from ecomplexity.calc_proximity import calc_continuous_proximity
from ecomplexity.ComplexityData import ComplexityData

def proximity(data, cols_input, presence_test="rca", val_errors_flag='coerce',
              rca_mcp_threshold=1, rpop_mcp_threshold=1, pop=None,
              continuous=False, asymmetric=False, verbose=1):
    """Wrapper function to calculate product proximity matrices

    Args:
        data: pandas df with cols 'time','loc','prod','val'
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
        continuous: Whether to consider correlation of every product pair (True)
            or product co-occurrence (False). *default* False.
        asymmetric: Whether to generate asymmetric proximity matrix (True) or
            symmetric (False). *default* False.
        verbose:  Outputs processing progress to stdout

    Returns:
        pandas df with proximity values for every product pair
    """

    cdata = ComplexityData(data, cols_input, val_errors_flag)

    output_list = []

    time_data = cdata.data.index.unique("time")
    if verbose > 0:
        time_data = tqdm.tqdm(cdata.data.index.unique("time"))

    # Iterate over time stamps
    for t in time_data:
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

        # Calculate proximity
        if continuous==False:
            prox_mat = calc_discrete_proximity(cdata.mcp_t, cdata.ubiquity_t,
                                               asymmetric)
        elif continuous==True and presence_test=="rpop":
            prox_mat = calc_continuous_proximity(cdata.rpop_t, cdata.ubiquity_t)
        elif continuous==True and presence_test!="rpop":
            prox_mat = calc_continuous_proximity(cdata.rca_t, cdata.ubiquity_t)

        # Reshape as df
        output_index = pd.MultiIndex.from_product([cdata.data_t.index.levels[1],
                                                   cdata.data_t.index.levels[1]])
        output = pd.DataFrame(data={'proximity':prox_mat.ravel()},
                              index=output_index)
        output['time'] = t
        output_list.append(output)

    output = pd.concat(output_list)

    # Remove entries for product's proximity with itself
    output = output.reset_index()
    output.columns = ['prod1','prod2','proximity','time']
    output = output[['time','prod1','prod2','proximity']]
    output = output[output.prod1!=output.prod2]

    # Rename based on original product column name
    output = output.rename(columns={'prod1':cols_input['prod']+'_1',
                                    'prod2':cols_input['prod']+'_2',
                                    'time':cols_input['time']})

    return(output)
