
# coding: utf-8

# In[8]:


# %reset
# Import libraries
import os
import numpy as np
import pandas as pd


# In[9]:


# Read example dataset
data = pd.read_csv('data/raw/year_origin_hs92_4.tsv',sep='\t')
data = data.drop(columns=['export_rca','import_rca','import_val'])
data.head()


# In[10]:


# Identify variables of interest and rename columns accordingly
cols_input = {'time':'year','loc':'origin','prod':'hs92','val':'export_val'}
cols_default = {'time':'time','loc':'loc','prod':'prod','val':'val'}
cols_map = {k:(cols_input[k] if k in cols_input else cols_default[k]) for k in cols_default}
cols_map_inv = {v:k for k,v in cols_map.items()}
data = data.rename(columns=cols_map_inv)
data = data[['time','loc','prod','val']]
data.head()

