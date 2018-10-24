#!/usr/bin/env python
# coding: utf-8

# In[352]:


get_ipython().run_line_magic('reset', '')
# Import libraries
import os
import numpy as np
import pandas as pd
import warnings


# In[353]:


# Read example dataset
data = pd.read_csv('data/raw/year_origin_hs92_4.tsv',sep='\t')
data.drop(columns=['export_rca','import_rca','import_val'], inplace=True)
data.head()


# In[354]:


# Get user input for variables of interest and rename columns accordingly
cols_input = {'time':'year','loc':'origin','prod':'hs92','val':'export_val'}
val_errors_flag_input = 'coerce' # Options: 'coerce','raise','ignore'

cols_default = {'time':'time','loc':'loc','prod':'prod','val':'val'}
cols_map = {k:(cols_input[k] if k in cols_input else cols_default[k]) for k in cols_default}
cols_map_inv = {v:k for k,v in cols_map.items()}
data.rename(columns=cols_map_inv, inplace=True)
data = data[['time','loc','prod','val']]

data.val = pd.to_numeric(data.val, errors=val_errors_flag_input)

data.set_index(['time','loc','prod'],inplace=True)

if data.val.isnull().values.any():
    warnings.warn('NaN\ value(s) present, coercing to zero(es)')
    data.val.fillna(0,inplace=True)

dups = data.index.duplicated()
if dups.sum()>0:
    warnings.warn('Duplicate values exist, keeping the first occurrence')
    data = data[~data.index.duplicated()]

data.head()


# In[355]:


# Create pandas dataframe with all possible combinations of values
time_vals = data.index.levels[0]
loc_vals = data.index.levels[1]
prod_vals = data.index.levels[2]
data_index = pd.MultiIndex.from_product(data.index.levels,names=data.index.names)
data = data.reindex(data_index, fill_value=0)


# In[356]:


# Calculate RCA numpy array
time_n_vals = len(data.index.levels[0])
loc_n_vals = len(data.index.levels[1])
prod_n_vals = len(data.index.levels[2])
data_np = data.values.reshape((time_n_vals,loc_n_vals,prod_n_vals))

num = (data_np/data_np.sum(axis=1)[:,np.newaxis,:])
loc_total = data_np.sum(axis=2)[:,:,np.newaxis]
world_total = loc_total.sum(axis=1)[:,np.newaxis,:]
den = loc_total/world_total
data_rca = num/den

data_rca.shape


# In[357]:


# Calculate MCP matrix
rca_mcp_threshold_input = 1

mcp = data_rca
mcp = np.nan_to_num(mcp)
mcp[data_rca>=rca_mcp_threshold_input] = 1
mcp[data_rca<rca_mcp_threshold_input] = 0
mcp.shape


# **Calculating complexity**
# 
# For calculating complexity, note that:
# 
# \begin{align}
#     \sum_{p} M_{cp} M_{c^{'} p} = M_{cp} \cdot M_{c^{'} p}
# \end{align}
# 
# In the 3d array, `mcp @ mcp.transpose(0,2,1)` recreates the same dot product behavior, but for each year separately.

# In[358]:


# Calculate complexity and other vars
diversity = mcp.sum(axis=2)
ubiquity = mcp.sum(axis=1)

mcp1 = mcp/diversity[:,:,np.newaxis]
mcp2 = mcp/ubiquity[:,np.newaxis,:]

mcp1[np.isnan(mcp1)] = 0
mcp2[np.isnan(mcp2)] = 0

Mcc = mcp1 @ mcp2.transpose(0,2,1)
Mpp = mcp1.transpose(0,2,1) @ mcp2


# In[359]:


Mcc.shape


# In[360]:


Mpp.shape


# In[361]:


def calculate_complexity(m_tilde):
    eigvals, eigvecs = np.linalg.eig(m_tilde)
    eigvecs = np.real(eigvecs)
    # Get eigenvector corresponding to second largest eigenvalue
    eig_index = eigvals.argsort(axis=1)[:,-2]
    # Fancy indexing to get complexity for each year
    complexity = eigvecs[np.arange(eigvecs.shape[0]),:,eig_index]
    return(complexity)


# In[362]:


def sign(k, kx_0):
    return(2 * int(np.corrcoef(k, kx_0)[0,1] > 0) - 1)


# In[382]:


def normalize(v):
    return (v - v.mean(axis=1)[:,np.newaxis]) / v.std(axis=1)[:,np.newaxis]


# In[383]:


kp = calculate_complexity(Mpp)
kc = calculate_complexity(Mcc)
print(kp.shape)
print(kc.shape)


# In[384]:


eci = normalize(sign(kc, diversity) * kc)
pci = normalize(sign(kp, ubiquity) * kp)

