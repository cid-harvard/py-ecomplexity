
# coding: utf-8

# In[21]:


get_ipython().run_line_magic('reset', '')


# In[22]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[23]:


# Import libraries
import os
import sys
sys.path.append('/Users/shg309/Dropbox/Education/hks_cid_growth_lab/misc/ecomplexity/')
import numpy as np
import pandas as pd
import warnings
from ecomplexity import ComplexityData


# In[24]:


# Read example dataset
data = pd.read_csv('data/raw/year_origin_hs92_4.tsv',sep='\t')
data.drop(columns=['export_rca','import_rca','import_val'], inplace=True)
data.head()


# In[33]:


check = ComplexityData(data,{'time':'year','loc':'origin','prod':'hs92','val':'export_val'})


# In[44]:


data.year.unique()


# In[50]:


random_pop_df = pd.DataFrame({'year':np.arange(1995,2015),'pop':np.random.randint(200000,2e6,len(data.year.unique()))})
rpop = check.calculate_rpop()

