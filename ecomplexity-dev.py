#!/usr/bin/env python
# coding: utf-8

# In[150]:


get_ipython().run_line_magic('reset', '')
# Import libraries
import os
import numpy as np
import pandas as pd
import warnings


# In[151]:


# Read example dataset
data = pd.read_csv('data/raw/year_origin_hs92_4.tsv',sep='\t')
data.drop(columns=['export_rca','import_rca','import_val'], inplace=True)
data.head()


# In[152]:


# Get user input for variables of interest and rename columns accordingly
cols_input = {'time':'year','loc':'origin','prod':'hs92','val':'export_val'}

cols_default = {'time':'time','loc':'loc','prod':'prod','val':'val'}
cols_map = {k:(cols_input[k] if k in cols_input else cols_default[k]) for k in cols_default}
cols_map_inv = {v:k for k,v in cols_map.items()}
data.rename(columns=cols_map_inv, inplace=True)
data = data[['time','loc','prod','val']]

data.val = data.val.astype(float)

data.set_index(['time','loc','prod'],inplace=True)

if data.val.isnull().values.any():
    warnings.warn('NaN\ value(s) present, coercing to zero(es)')
    data.val.fillna(0,inplace=True)

dups = data.index.duplicated()
if dups.sum()>0:
    warnings.warn('Duplicate values exist, keeping the first occurrence')
    data = data[~data.index.duplicated()]

data.head()


# In[153]:


# Create pandas dataframe with all possible combinations of values
time_vals = data.index.levels[0]
loc_vals = data.index.levels[1]
prod_vals = data.index.levels[2]
data_index = pd.MultiIndex.from_product(data.index.levels,names=data.index.names)
data = data.reindex(data_index, fill_value=0)


# In[154]:


# Calculate RCA numpy array
time_n_vals = len(data.index.levels[0])
loc_n_vals = len(data.index.levels[1])
prod_n_vals = len(data.index.levels[2])
data_np = data.values.reshape((time_n_vals,loc_n_vals,prod_n_vals))

num = (data_np/data_np.sum(axis=1)[:,np.newaxis,:])
den = data_np.sum(axis=2)[:,:,np.newaxis]/data_np.sum(axis=2)[:,:,np.newaxis].sum(axis=1)[:,np.newaxis,:]
data_rca = num/den

data_rca.shape


# In[20]:


# Calculate MCP matrix
rca_mcp_threshold_input = 1

mcp = data_rca
mcp[data_rca>=rca_mcp_threshold_input] = 1
mcp[data_rca<rca_mcp_threshold_input | data_rca.isnan()] = 0
mcp.shape


# In[ ]:


def np_ecomplexity(data_np):
    data['mcp'] = (data.val>=rca_input).astype(int)
    
    # Get number of countries and products
    ncx, npx = mcp.shape

    # Calculate diversity and ubiquity matrices
    kc0 = mcp @ np.full((npx, npx), 1)
    kp0 = np.full((ncx, ncx), 1) @ mcp

    kp0_1d = cpy.sum().values
    kc0_1d = cpy.T.sum().values
    


# In[21]:


data.describe()


# In[ ]:




