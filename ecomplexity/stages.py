import pandas as pd
from siuba import _, inner_join, left_join, full_join, group_by, summarize, filter, right_join
from siuba.experimental.pd_groups import fast_mutate, fast_filter, fast_summarize
import numpy as np

from ecomplexity.calc_proximity import calc_discrete_proximity


def Rca_gen(trade_data: pd.DataFrame, t=None) -> pd.DataFrame:
    """STEP 1 - helper function

    Parameters
    ----------
    trade_data : pd.DataFrame
        contains columns: ("c", "naics", "year", "export_value")
    t : int, optional
        value for year, if we are just calculating for one particular year,
        by default None

    Returns
    -------
    pd.DataFrame
        country x naics
    """

    if t is not None:
        df = trade_data >> filter(_.year == t)
    else:
        df = trade_data

    sum_p_X_cp = (df 
        >> group_by(_.c) # group by year: TODO potentially
        >> fast_summarize(sum_p_X_cp = _.export_value.sum()) 
    )

    sum_c_X_cp = (df 
        >> group_by(_.naics) # group by year: TODO potentially
        >> fast_summarize(sum_c_X_cp = _.export_value.sum())
    )

    df = inner_join(df, sum_c_X_cp, on = 'naics') # add year to calculate all years at a single time (TODO potentially)
    df = inner_join(df, sum_p_X_cp, on = 'c') # add year to calculate all years at a single time (TODO potentially)

    sum_cp_X_cp = sum(df['export_value'])

    df['Rca'] = (df['export_value'] / df['sum_c_X_cp']) / (df['sum_p_X_cp'] / sum_cp_X_cp)

    # creates matrix. Enables indexing by loc in the following way: rca_mat.loc[country_value, naics_value]
    rca_mat = df.pivot(index ='c', columns ='naics', values = 'Rca').fillna(0)

    return rca_mat

def Mcp_base_gen(trade_data: pd.DataFrame, threshold: float, t=None) -> pd.DataFrame:
    """STEP 1 - HELPER FUNCTION

    Parameters
    ----------
    trade_data : pd.DataFrame
        contains columns: ("c", "naics", "year", "export_value")
    threshold : float
        threshold for binarizing entries in matrix
    t : int, optional
        value for year, if we are just calculating for one particular year, 
        by default None

    Returns
    -------
    pd.DataFrame
        country x naics
        matrix of 0,1's - depends on threshold
    """
    Rca_mat = Rca_gen(trade_data, t=t)

    Rca_mat[Rca_mat >= threshold] = 1
    Rca_mat[Rca_mat < threshold] = 0

    Mcp_mat = Rca_mat

    return Mcp_mat


def Mcp_post_agg(Mcp_dict: dict, post_agg_t_fraction: float) -> pd.DataFrame:
    """STEP 1 - HELPER FUNCTION

    Parameters
    ----------
    Mcp_dict : dictionary
        dictionary with keys being years, values being Mcp for the corresponding year
    post_agg_t_fraction : float
        fraction of entries that must be 1, for the binarized value to be 1 in the post-aggregated Mcp

    Returns
    -------
    pd.DataFrame
        binarized Mcp according to post_agg_t_fraction
    """
    # add up all elements in dict
    Mcp_sum = 0
    for Mcp in Mcp_dict.values():
        Mcp_sum = Mcp_sum + Mcp # TODO: check how += and = ... + ... differ

    # apply thresholding
    Mcp_sum = Mcp_sum.fillna(0)
    
    return Mcp_sum.applymap(lambda x: 1 if x/len(Mcp_dict) > post_agg_t_fraction else 0)


def Mcp_gen(trade_data: pd.DataFrame, how='single_t', t=None, threshold=1.0, 
            post_agg_t_slice=None, post_agg_t_fraction=None):
    """
    STEP 1

    Parameters
    ----------
    trade_data : pd.DataFrame
        input data. Must have at minimum the four following columns: c, naics, year, export_value
    how : str, optional
        must be element of ['single_t', 'pre_aggregate', 'post_aggregate'], by default 'single_t'
        => if 'single_t', t must be specified
        => if 'pre_aggregate', no other parameters should be specified
        => if 'post_aggregate', post_agg_t_slice and post_agg_t_fraction must be specified
    t : int, optional
        value for year, if we are just calculating for one particular year, 
        by default None
    threshold : float, optional
        threshold for binarizing variables in matrix, by default 1.0
    post_agg_t_slice: int, optional
        desired width of post-aggregation time slice
        e.g. 5 would split years 2000-2020 into 2001-2005, 2006-2010, 2011-2015, 2016-2020
    post_agg_t_fraction: float, optional
        fraction of entries that must be 1, for the binarized value to be 1 in the post-aggregated Mcp


    Returns
    -------
    pd.DataFrame
        country x naics
    OR dictionary
        if how = 'post_aggregate'
    """
    if how == 'single_t':
        return Mcp_base_gen(trade_data, threshold, t=t)

    elif how == 'pre_aggregate':
        trade_data = (trade_data
            >> group_by(_.c, _.naics)
            >> fast_summarize(export_value = _.export_value.sum())
        )

        return Mcp_base_gen(trade_data, threshold)

    else:
        # post aggregation

        Mcp_dict = dict() # contains Mcp matrices for all years

        for t in set(trade_data['year']):
            Mcp_dict[t] = Mcp_base_gen(trade_data, threshold=threshold, t=t) 

        time_list = list(set(trade_data['year']))

        # array_split allows indices_or_sections to be an integer that does not equally 
        # divide the axis. For an array of length l that should be split into n sections, 
        # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
        
        splits = np.array_split(time_list, int(len(time_list)/post_agg_t_slice))

        Mcp_post_agg_dict = dict()

        for time_ar in splits:
            Mcp_dict_slice = {key: value for (key,value) in Mcp_dict.items() if key in time_ar}
            Mcp_post_agg_dict[time_ar[0]] = Mcp_post_agg(Mcp_dict_slice, post_agg_t_fraction)
        
        return Mcp_post_agg_dict
    

def PHIpp_gen(Mcp: pd.DataFrame) -> pd.DataFrame:
    """STEP 3

    Parameters
    ----------
    Mcp : pd.DataFrame
        country x naics

    Returns
    -------
    pd.DataFrame
        naics x naics
    """
    ubiquity = np.nansum(Mcp, axis=0)

    return calc_discrete_proximity(Mcp, ubiquity, asymmetric=False)


def Cca_gen(Mcp, Ppa):
    """STEP 5

    Parameters
    ----------
    Mcp : pd.DataFrame
        country x naics
    Ppa : pd.DataFrame
        naics x capability

    Returns
    -------
    Cca: pd.DataFrame
        country x capability matrix
    """
    Cca_raw = Mcp @ Ppa

    return Cca_raw.applymap(lambda x: 1 if x > 0 else 0)


def MUcp_gen(Ppa: pd.DataFrame, Cca: pd.DataFrame) -> pd.DataFrame:
    """STEP 6

    Parameters
    ----------
    Ppa : pd.DataFrame
        naics x capability
    Cca : pd.DataFrame
        country x capability

    Returns
    -------
    Mucp: pd.DataFrame
        country x naics
    """

    return Cca @ Ppa.T
    

def Gpp_gen(Ppa: pd.DataFrame) -> pd.DataFrame:
    """STEP 7

    Parameters
    ----------
    Ppa : pd.DataFrame
        naics x capability
    
    Returns
    -------
    Gpp: pd.DataFrame
        naics x naics
    """

    numerator = Ppa @ Ppa.T # naics x naics
    denominator = Ppa.mean(axis=1).T

    return numerator / denominator

def k0_helper(data_mat, threshold=0):
    Ppa_filtered = data_mat.applymap(lambda x: 1 if x > threshold else 0)
    return (Ppa_filtered.sum(axis=0).to_frame().rename(columns={0: 'sm'}), 
            data_mat.sum(axis=1).to_frame().rename(columns={0: 'sm'}))

def k0_gen(Ppa, Cca, Mcp, Ppa_threshold=0, Cca_threshold=0, Mcp_threshold=0):
    capability_generality, industry_span = k0_helper(data_mat=Ppa, threshold=Ppa_threshold)
    capability_ubiquity, country_completeness = k0_helper(data_mat=Cca, threshold=Cca_threshold)
    industry_ubiquity, country_diversity = k0_helper(data_mat=Mcp, threshold=Mcp_threshold)
    return (capability_generality, 
            capability_ubiquity,
            industry_span, 
            industry_ubiquity,
            country_completeness, 
            country_diversity)

def k1_helper(melted_mat, join_mat, join_on, group_on):
    return (melted_mat 
            >> inner_join(_, join_mat, on = join_on)
            >> group_by(group_on)
            >> fast_summarize(avg_metric = _.sm.mean())
        )   

def k1_gen(Ppa, Cca, Mcp, Ppa_threshold=0, Cca_threshold=0, Mcp_threshold=0):
    (capability_generality, 
        capability_ubiquity,
        industry_span, 
        industry_ubiquity,
        country_completeness, 
        country_diversity) = k0_gen(Ppa, Cca, Mcp, Ppa_threshold, Cca_threshold, Mcp_threshold)
    
    Mcp_melted = Mcp.reset_index().melt(id_vars='c', var_name='naics', value_name = 'x') >> filter(_.x != 0) 
    Cca_melted = Cca.reset_index().melt(id_vars='c', var_name='capability', value_name = 'x') >> filter(_.x != 0) 
    Ppa_melted = Ppa.reset_index().melt(id_vars='naics', var_name='capability', value_name = 'x') >> filter(_.x >= Ppa_threshold) 

    avg_diversity_c_ind = k1_helper(melted_mat = Mcp_melted, join_mat=country_diversity, join_on="c", group_on="naics")
    avg_ubiquity_ind_c = k1_helper(melted_mat = Mcp_melted, join_mat=industry_ubiquity, join_on="naics", group_on="c")
    avg_ubiquity_cap_c = k1_helper(melted_mat = Cca_melted, join_mat=capability_ubiquity, join_on="capability", group_on="c")
    avg_completeness_c_cap = k1_helper(melted_mat = Cca_melted, join_mat=country_completeness, join_on="c", group_on="capability")

    avg_generality_cap_ind = k1_helper(melted_mat = Ppa_melted, join_mat=capability_generality, join_on="capability", group_on="naics")
    avg_ubiquity_cap_ind = k1_helper(melted_mat = Ppa_melted, join_mat=capability_ubiquity, join_on="capability", group_on="naics")
    avg_completeness_c_ind = k1_helper(melted_mat = Mcp_melted, join_mat=country_completeness, join_on="c", group_on="naics")
    
    avg_span_ind_c = k1_helper(melted_mat = Mcp_melted, join_mat=industry_span, join_on="naics", group_on="c")
    avg_generality_cap_c = k1_helper(melted_mat = Cca_melted, join_mat=capability_generality, join_on="capability", group_on="c")

    avg_span_ind_cap = k1_helper(melted_mat = Ppa_melted, join_mat=industry_span, join_on="naics", group_on="capability")
    avg_ubiquity_ind_cap = k1_helper(melted_mat = Ppa_melted, join_mat=industry_ubiquity, join_on="naics", group_on="capability")
    avg_diversity_c_cap = k1_helper(melted_mat = Cca_melted, join_mat=country_diversity, join_on="c", group_on="capability")


    return (avg_diversity_c_ind,
    avg_ubiquity_ind_c,
    avg_ubiquity_cap_c,
    avg_completeness_c_cap,
    avg_generality_cap_ind,
    avg_ubiquity_cap_ind,
    avg_completeness_c_ind,
    avg_span_ind_c,
    avg_generality_cap_c,
    avg_span_ind_cap,
    avg_ubiquity_ind_cap,
    avg_diversity_c_cap)