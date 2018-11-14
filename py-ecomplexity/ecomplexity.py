# Complexity calculations
import numpy as np
import pandas as pd
import warnings
import sys
from functools import wraps
import time
import datetime

# # Get user input
# cols_input = {'time':'year','loc':'origin','prod':'hs92','val':'export_val'}
# val_errors_flag_input = 'coerce' # Options: 'coerce','raise','ignore'
# rca_mcp_threshold_input = 1


def timethis(func):
    """Time a function of your choice"""
    @wraps(func)
    def wrapper(*arg, **kw):
        t = time.time()
        res = func(*arg, **kw)
        runtime = str(datetime.timedelta(seconds=(time.time() - t)))
        print("TIMING: function {} took {}.".format(func.__name__, runtime))
        return res
    return wrapper


class ComplexityData(object):
    """Calculate complexity and other related results

    Args:
        data: pandas dataframe containing production / trade data.
            Including variables indicating time, location, product and value
        cols_input: dict of column names for time, location, product and value.
            Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
        val_errors_flag_input: {'coerce','ignore','raise'}. Passed to pd.to_numeric
            *default* coerce.
        rca_mcp_threshold_input: numeric indicating threshold beyond which mcp is 1.
            *default* 1.

    Attributes:
        diversity: k_c,0
        ubiquity: k_p,0
        rca: RCA matrix
        eci: Economic complexity index
        pci: Product complexity index
    """

    def __init__(self, data, cols_input, val_errors_flag_input='coerce', rca_mcp_threshold_input=1):
        self.data = data
        self.rename_cols(cols_input)
        self.clean_data(val_errors_flag_input)
        self.create_full_df()
        self.calculate_rca_and_mcp(rca_mcp_threshold_input)
        self.diversity = np.nansum(self.mcp, axis=2)
        self.ubiquity = np.nansum(self.mcp, axis=1)

        Mcc, Mpp = self.calculate_Mcc_Mpp()

        kp = self.calculate_Kvec(Mpp)
        kc = self.calculate_Kvec(Mcc)

        self.eci = self.normalize(self.sign(kc, self.diversity) * kc)
        self.pci = self.normalize(self.sign(kp, self.ubiquity) * kp)

        self.reshape_output_to_data()

    @timethis
    def rename_cols(self, cols_input):
        # Rename cols
        cols_default = {'time': 'time', 'loc': 'loc',
                        'prod': 'prod', 'val': 'val'}
        cols_map = {k: (cols_input[k] if k in cols_input else cols_default[
                        k]) for k in cols_default}
        cols_map_inv = {v: k for k, v in cols_map.items()}
        self.data = self.data.rename(columns=cols_map_inv)
        self.data = self.data[['time', 'loc', 'prod', 'val']]

    @timethis
    def clean_data(self, val_errors_flag_input):
        # Make sure values are numeric
        self.data.val = pd.to_numeric(
            self.data.val, errors=val_errors_flag_input)
        self.data.set_index(['time', 'loc', 'prod'], inplace=True)
        if self.data.val.isnull().values.any():
            warnings.warn('NaN value(s) present, coercing to zero(es)')
            self.data.val.fillna(0, inplace=True)
        dups = self.data.index.duplicated()
        if dups.sum() > 0:
            warnings.warn(
                'Duplicate values exist, keeping the first occurrence')
            self.data = self.data[~self.data.index.duplicated()]

    @timethis
    def create_full_df(self):
        # Create pandas dataframe with all possible combinations of values
        data_index = pd.MultiIndex.from_product(
            self.data.index.levels, names=self.data.index.names)
        self.data = self.data.reindex(data_index, fill_value=0)

    @timethis
    def calculate_rca_and_mcp(self, rca_mcp_threshold_input):
        # Convert data into numpy array
        time_n_vals = len(self.data.index.levels[0])
        loc_n_vals = len(self.data.index.levels[1])
        prod_n_vals = len(self.data.index.levels[2])
        data_np = self.data.values.reshape(
            (time_n_vals, loc_n_vals, prod_n_vals))

        # Calculate RCA, disable dividebyzero errors
        with np.errstate(divide='ignore', invalid='ignore'):
            num = (data_np / np.nansum(data_np, axis=2)[:, :, np.newaxis])
            loc_total = np.nansum(data_np, axis=1)[:, np.newaxis, :]
            world_total = np.nansum(loc_total, axis=2)[:, :, np.newaxis]
            den = loc_total / world_total
            self.rca = num / den

        # Calculate MCP matrixs
        self.mcp = self.rca
        self.mcp = np.nan_to_num(self.mcp)
        self.mcp = np.where(self.mcp >= rca_mcp_threshold_input, 1, 0)

    def calculate_rpop(self, pop):
        """Find RPOP. WORK IN PROGRESS!

        Args:
            pop: A pandas df, with time, location and corresponding population,
                in that order

        Returns:
            rpop: numpy array with rpop values
        """
        # After constructing df with all combinations, convert data into ndarray
        time_n_vals = len(self.data.index.levels[0])
        loc_n_vals = len(self.data.index.levels[1])
        prod_n_vals = len(self.data.index.levels[2])
        data_np = self.data.values.reshape(
            (time_n_vals, loc_n_vals, prod_n_vals))

        pop.columns = ['time', 'loc', 'pop']
        pop = pop.reset_index(drop=True).set_index(['time', 'loc'])
        pop_index = pd.MultiIndex.from_product(
            [self.data.index.levels[0], self.data.index.levels[1]],
            names=['time', 'loc'])
        pop = pop.reindex(pop_index)
        time_n_vals_pop = len(pop.index.levels[0])
        loc_n_vals_pop = len(pop.index.levels[1])
        pop = pop.values.reshape((time_n_vals_pop, loc_n_vals_pop))

        with np.errstate(divide='ignore', invalid='ignore'):
            num = data_np / pop[:, :, np.newaxis]
            loc_total = np.nansum(data_np, axis=1)[:, np.newaxis, :]
            world_pop_total = np.nansum(pop, axis=1)[:, np.newaxis, np.newaxis]
            den = loc_total / world_pop_total
            rpop = num / den
        return(rpop)

    @timethis
    def calculate_Mcc_Mpp(self):

        with np.errstate(divide='ignore', invalid='ignore'):
            mcp1 = self.mcp / self.diversity[:, :, np.newaxis]
            mcp2 = self.mcp / self.ubiquity[:, np.newaxis, :]

        mcp1[np.isnan(mcp1)] = 0
        mcp2[np.isnan(mcp2)] = 0

        # These matrix multiplication lines are *very* slow
        Mcc = mcp1 @ mcp2.transpose(0, 2, 1)
        Mpp = mcp1.transpose(0, 2, 1) @ mcp2
        return(Mcc, Mpp)

    @timethis
    def reshape_output_to_data(self):
        diversity = self.diversity[:,:,np.newaxis].repeat(self.rca.shape[2], axis=2).ravel()
        ubiquity = self.ubiquity[:,np.newaxis,:].repeat(self.rca.shape[1], axis=1).ravel()
        eci = self.eci[:, :, np.newaxis].repeat(self.rca.shape[2], axis=2).ravel()
        pci = self.pci[:, np.newaxis, :].repeat(self.rca.shape[1], axis=1).ravel()
        output = pd.DataFrame.from_dict({'diversity': diversity,
                                         'ubiquity': ubiquity,
                                         'rca': self.rca.ravel(),
                                         'eci': eci,
                                         'pci': pci}).reset_index(drop=True)
        self.output = pd.concat([self.data.reset_index(), output], axis=1)

    @staticmethod
    @timethis
    def calculate_Kvec(m_tilde):
        eigvals, eigvecs = np.linalg.eig(m_tilde)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort(axis=1)[:, -2]
        # Fancy indexing to get complexity for each year
        Kvec = eigvecs[np.arange(eigvecs.shape[0]), :, eig_index]
        return(Kvec)

    @staticmethod
    def sign(k, kx_0):
        return(2 * int(np.corrcoef(k, kx_0)[0, 1] > 0) - 1)

    @staticmethod
    def normalize(v):
        return(v - v.mean(axis=1)[:, np.newaxis]) / v.std(axis=1)[:, np.newaxis]
