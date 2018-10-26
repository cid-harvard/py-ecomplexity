# Complexity calculations
import numpy as np
import pandas as pd
import warnings

# # Get user input
# cols_input = {'time':'year','loc':'origin','prod':'hs92','val':'export_val'}
# val_errors_flag_input = 'coerce' # Options: 'coerce','raise','ignore'
# rca_mcp_threshold_input = 1

class ComplexityData(object):
    """Calculate complexity and other related results

    Args:
        data: pandas dataframe containing production / trade data, including variables
            indicating time, location, product and value
        cols_input: dict indicating column names for cols indicating time, location,
            product and value. Example: {'time':'year', 'loc':'origin', 'prod':'hs92',
            'val':'export_val'}
        val_errors_flag_input: {'coerce','ignore','raise'}. *default* coerce.
            Passed to pd.to_numeric
        rca_mcp_threshold_input: numeric indicating threshold beyond which mcp is 1. *default* 1.

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
        self.diversity = self.mcp.sum(axis=2)
        self.ubiquity = self.mcp.sum(axis=1)

        Mcc, Mpp = self.calculate_Mcc_Mpp()

        kp = self.calculate_Kvec(Mpp)
        kc = self.calculate_Kvec(Mcc)

        self.eci = self.normalize(self.sign(kc, self.diversity) * kc)
        self.pci = self.normalize(self.sign(kp, self.ubiquity) * kp)

    def rename_cols(self, cols_input):
        # Rename cols
        cols_default = {'time':'time','loc':'loc','prod':'prod','val':'val'}
        cols_map = {k:(cols_input[k] if k in cols_input else cols_default[k]) for k in cols_default}
        cols_map_inv = {v:k for k,v in cols_map.items()}
        self.data = self.data.rename(columns=cols_map_inv)
        self.data = self.data[['time','loc','prod','val']]

    def clean_data(self, val_errors_flag_input):
        # Make sure values are numeric
        self.data.val = pd.to_numeric(self.data.val, errors=val_errors_flag_input)
        self.data.set_index(['time','loc','prod'],inplace=True)
        if self.data.val.isnull().values.any():
            warnings.warn('NaN\ value(s) present, coercing to zero(es)')
            self.data.val.fillna(0,inplace=True)
        dups = self.data.index.duplicated()
        if dups.sum()>0:
            warnings.warn('Duplicate values exist, keeping the first occurrence')
            self.data = self.data[~self.data.index.duplicated()]

    def create_full_df(self):
        # Create pandas dataframe with all possible combinations of values
        time_vals = self.data.index.levels[0]
        loc_vals = self.data.index.levels[1]
        prod_vals = self.data.index.levels[2]
        data_index = pd.MultiIndex.from_product(self.data.index.levels,names=self.data.index.names)
        self.data = self.data.reindex(data_index, fill_value=0)

    def calculate_rca_and_mcp(self, rca_mcp_threshold_input):
        # Calculate RCA numpy array
        time_n_vals = len(self.data.index.levels[0])
        loc_n_vals = len(self.data.index.levels[1])
        prod_n_vals = len(self.data.index.levels[2])
        data_np = self.data.values.reshape((time_n_vals,loc_n_vals,prod_n_vals))

        num = (data_np/data_np.sum(axis=1)[:,np.newaxis,:])
        loc_total = data_np.sum(axis=2)[:,:,np.newaxis]
        world_total = loc_total.sum(axis=1)[:,np.newaxis,:]
        den = loc_total/world_total
        self.rca = num/den

        # Calculate MCP matrix
        self.mcp = self.rca
        self.mcp = np.nan_to_num(self.mcp)
        self.mcp[self.rca>=rca_mcp_threshold_input] = 1
        self.mcp[self.rca<rca_mcp_threshold_input] = 0

    def calculate_rpop(self, pop):
        """Find RPOP. WORK IN PROGRESS!

        Args:
            pop: A pandas df, with time, location and corresponding population,
                in that order

        Returns:
            rpop: numpy array with rpop values
        """
        time_n_vals = len(self.data.index.levels[0])
        loc_n_vals = len(self.data.index.levels[1])
        prod_n_vals = len(self.data.index.levels[2])
        data_np = self.data.values.reshape((time_n_vals,loc_n_vals,prod_n_vals))

        pop.columns = ['time','loc','pop']
        pop = pop.reset_index().set_index(['time','loc'])
        time_n_vals = len(self.pop.index.levels[0])
        loc_n_vals = len(self.pop.index.levels[1])
        pop = pop.values.reshape((time_n_vals,loc_n_vals))

        num = data_np / pop[:,:,np.newaxis]
        time_total = data_np.sum(axis=1)[:,np.newaxis,:]
        world_pop_total = pop.sum(axis=1)[:,np.newaxis,np.newaxis]
        den = time_total/world_pop_total
        rpop = num/den
        return(rpop)

    def calculate_Mcc_Mpp(self):
        mcp1 = self.mcp/self.diversity[:,:,np.newaxis]
        mcp2 = self.mcp/self.ubiquity[:,np.newaxis,:]

        mcp1[np.isnan(mcp1)] = 0
        mcp2[np.isnan(mcp2)] = 0

        Mcc = mcp1 @ mcp2.transpose(0,2,1)
        Mpp = mcp1.transpose(0,2,1) @ mcp2
        return(Mcc,Mpp)

    @staticmethod
    def calculate_Kvec(m_tilde):
        eigvals, eigvecs = np.linalg.eig(m_tilde)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort(axis=1)[:,-2]
        # Fancy indexing to get complexity for each year
        Kvec = eigvecs[np.arange(eigvecs.shape[0]),:,eig_index]
        return(Kvec)

    @staticmethod
    def sign(k, kx_0):
        return(2 * int(np.corrcoef(k, kx_0)[0,1] > 0) - 1)

    @staticmethod
    def normalize(v):
        return (v - v.mean(axis=1)[:,np.newaxis]) / v.std(axis=1)[:,np.newaxis]
