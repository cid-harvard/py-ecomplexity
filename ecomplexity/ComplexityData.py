# Complexity calculations
import numpy as np
import pandas as pd
import warnings
import sys
from functools import wraps
import time
import datetime


class ComplexityData(object):
    """Calculate complexity and other related results

    Args:
        data: pandas dataframe containing production / trade data.
            Including variables indicating time, location, product and value
        cols_input: dict of column names for time, location, product and value.
            Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
        val_errors_flag: {'coerce','ignore','raise'}. Passed to pd.to_numeric
            *default* coerce.

    Attributes:
        data: clean data with standardized column names
    """

    def __init__(self, data, cols_input, val_errors_flag):
        self.data = data.copy()
        self.cols_input = cols_input

        # Standardize column names based on input
        self.rename_cols()

        # Clean data to handle NA's and such
        self.clean_data(val_errors_flag)

    def rename_cols(self):
        """Standardize column names"""
        cols_map_inv = {v: k for k, v in self.cols_input.items()}
        self.data = self.data.rename(columns=cols_map_inv)
        self.data = self.data[["time", "loc", "prod", "val"]]

    def clean_data(self, val_errors_flag_input):
        """Clean data to remove non-numeric values, handle NA's and duplicates"""
        # Make sure values are numeric
        self.data.val = pd.to_numeric(self.data.val, errors=val_errors_flag_input)
        self.data.set_index(["time", "loc", "prod"], inplace=True)
        if self.data.val.isnull().values.any():
            warnings.warn("NaN value(s) present, coercing to zero(es)")
            self.data.val.fillna(0, inplace=True)

        # Remove duplicates
        dups = self.data.index.duplicated()
        if dups.sum() > 0:
            warnings.warn("Duplicate values exist, keeping the first occurrence")
            self.data = self.data[~self.data.index.duplicated()]

    def create_full_df(self, t):
        """Rectangularize, but remove rows with diversity or ubiquity zero

        Rows with zero diversity / ubiquity lead to ZeroDivision errors and
        incorrect values during normalization
        """
        self.t = t
        self.data_t = self.data.loc[t].copy()
        # Check for zero diversity and ubiquity
        val_diversity_check = (
            self.data_t.reset_index().groupby(["loc"])["val"].sum().reset_index()
        )
        val_ubiquity_check = (
            self.data_t.reset_index().groupby(["prod"])["val"].sum().reset_index()
        )
        val_diversity_check = val_diversity_check[val_diversity_check.val != 0]
        val_ubiquity_check = val_ubiquity_check[val_ubiquity_check.val != 0]
        # Remove locations and products with zero diversity and ubiquity respectively
        self.data_t = self.data_t.reset_index()
        self.data_t = self.data_t.merge(
            val_diversity_check[["loc"]], on="loc", how="right"
        )
        self.data_t = self.data_t.merge(
            val_ubiquity_check[["prod"]], on="prod", how="right"
        )
        self.data_t.set_index(["loc", "prod"], inplace=True)
        # Create full dataframe with all combinations of locations and products
        data_index = pd.MultiIndex.from_product(
            self.data_t.index.levels, names=self.data_t.index.names
        )
        self.data_t = self.data_t.reindex(data_index, fill_value=0)

    def calculate_rca(self):
        """Calculate RCA"""
        # Convert data into numpy array
        loc_n_vals = len(self.data_t.index.levels[0])
        prod_n_vals = len(self.data_t.index.levels[1])
        data_np = self.data_t.values.reshape((loc_n_vals, prod_n_vals))

        # Calculate RCA, disable dividebyzero errors
        with np.errstate(divide="ignore", invalid="ignore"):
            num = data_np / np.nansum(data_np, axis=1)[:, np.newaxis]
            loc_total = np.nansum(data_np, axis=0)[np.newaxis, :]
            world_total = np.nansum(loc_total, axis=1)[:, np.newaxis]
            den = loc_total / world_total
            self.rca_t = num / den

    def calculate_rpop(self, pop, t):
        """Calculate RPOP"""
        # After constructing df with all combinations, convert data into ndarray
        loc_n_vals = len(self.data_t.index.levels[0])
        prod_n_vals = len(self.data_t.index.levels[1])
        data_np = self.data_t.values.reshape((loc_n_vals, prod_n_vals))

        # Read population data for selected year
        pop_t = pop[pop[self.cols_input["time"]] == t].copy()
        pop_t.columns = ["time", "loc", "pop"]
        pop_t = pop_t.drop(columns="time")

        pop_t = pop_t.reset_index(drop=True).set_index("loc")
        pop_index = self.data_t.index.unique("loc")
        pop_t = pop_t.reindex(pop_index)
        pop_t = pop_t.values
        assert (
            pop_t.shape[0] == data_np.shape[0]
        ), f"Year {t}: Trade and population data have to be available for the same countries / locations"

        num = data_np / pop_t
        loc_total = np.nansum(data_np, axis=0)[np.newaxis, :]
        world_pop_total = np.nansum(pop_t)

        den = loc_total / world_pop_total
        rpop = num / den
        self.rpop_t = rpop

    def calculate_mcp(
        self, rca_mcp_threshold_input, rpop_mcp_threshold_input, presence_test, pop, t
    ):
        """Calculate MCP based on RCA / RPOP / both"""

        def convert_to_binary(x, threshold):
            x = np.nan_to_num(x)
            x = np.where(x >= threshold, 1, 0)
            return x

        if presence_test == "rca":
            self.mcp_t = convert_to_binary(self.rca_t, rca_mcp_threshold_input)

        elif presence_test == "rpop":
            self.calculate_rpop(pop, t)
            self.mcp_t = convert_to_binary(self.rpop_t, rpop_mcp_threshold_input)

        elif presence_test == "both":
            self.calculate_rpop(pop, t)
            self.mcp_t = convert_to_binary(
                self.rca_t, rca_mcp_threshold_input
            ) + convert_to_binary(self.rpop_t, rpop_mcp_threshold_input)

    def calculate_manual_mcp(self):
        """If pre-computed MCP supplied, reshape"""
        # Convert data into numpy array
        loc_n_vals = len(self.data_t.index.levels[0])
        prod_n_vals = len(self.data_t.index.levels[1])
        data_np = self.data_t.values.reshape((loc_n_vals, prod_n_vals))

        self.mcp_t = data_np
