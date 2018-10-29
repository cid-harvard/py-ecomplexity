
# coding: utf-8

# In[94]:


get_ipython().run_line_magic('reset', '')


# In[95]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[96]:


# Import libraries
import os
import sys
sys.path.append('/Users/shg309/Dropbox/Education/hks_cid_growth_lab/misc/ecomplexity/')
import numpy as np
import pandas as pd
import warnings
from ecomplexity import ComplexityData


# In[97]:


# Read example dataset
ROOT = '/Users/shg309/Dropbox/Education/hks_cid_growth_lab/misc/ecomplexity/'
data = pd.read_csv(ROOT + 'data/raw/year_origin_hs92_4.tsv',sep='\t')
data.drop(columns=['export_rca','import_rca','import_val'], inplace=True)
data = data[~data.origin.str.startswith('xx')]
data.head()


# In[98]:


check = ComplexityData(data,{'time':'year','loc':'origin','prod':'hs92','val':'export_val'})


# In[99]:


# Create pop df to test rpop
from itertools import product

random_pop_df = pd.DataFrame(list(product(np.arange(2000,2010), ['aus','usa','jam'])))
random_pop_df.columns = ['time','loc']
random_pop_df['pop'] = np.random.randint(200000,2e6,len(random_pop_df))
random_pop_df.head()


# In[100]:


# Test rpop
np.nansum(check.calculate_rpop(random_pop_df))


# In[101]:


np.where(~np.isnan(pop))


# In[ ]:


### Temp
a = pd.DataFrame(list(product(np.arange(2000,2010), ['aus','usa','jam'], ['prod1','prod2'])))
a.columns = ['time','loc','prod']
a['val'] = np.random.randint(200000,2e6,len(a))
a = a[~((a.time==2002) & (a['loc']=='jam'))]
a = a[~((a.time==2003) & (a['loc']=='aus'))]
a = a.set_index(['time','loc','prod'])
# a


# In[ ]:


## Temp
index_df = data.head(n=100000).copy().reset_index(drop=True)     .rename(columns={'year':'time','origin':'loc'})     .set_index(['time','loc'])
pop_index = pd.MultiIndex.from_product(
            [index_df.index.levels[0], index_df.index.levels[1]],
            names=['time', 'loc'])
random_pop_df.reindex(index=pop_index)
index_df


# In[ ]:


### Temp Testing ###
a = np.arange(24).reshape(2,3,4)


# In[ ]:


a = data.head()
a.head()               

