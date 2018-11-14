
# coding: utf-8

# In[75]:


get_ipython().run_line_magic('reset', '')


# In[76]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[87]:


# Import libraries
import os
import sys
# sys.path.append('/Users/shg309/Dropbox/Education/hks_cid_growth_lab/misc/ecomplexity/')
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from ecomplexity import ComplexityData


# In[78]:


# Read example dataset
data = pd.read_csv('data/raw/year_origin_hs92_4.tsv',sep='\t')
data.drop(columns=['export_rca','import_rca','import_val'], inplace=True)
data = data[~data.origin.str.startswith('xx')]
data = data[data.export_val.notnull() & data.export_val!=0]
data = data[data.year.isin(list(range(1995,2001)))]
data.head()


# In[79]:


data.describe(include='all')


# In[80]:


cdata = ComplexityData(data,{'time':'year','loc':'origin','prod':'hs92','val':'export_val'})


# In[81]:


# Create pop df to test rpop
from itertools import product

random_pop_df = pd.DataFrame(list(product(np.arange(2000,2010), ['aus','usa','jam'])))
random_pop_df.columns = ['time','loc']
random_pop_df['pop'] = np.random.randint(200000,2e6,len(random_pop_df))
random_pop_df.head()


# In[82]:


# Test rpop
np.nansum(cdata.calculate_rpop(random_pop_df))


# In[83]:


cdata.output.head()


# Note that we cannot test ECI and PCI against stata output because they are normalized values, so if there's anything off, we can't tell where the difference stems from. To compare, we generate ECI / PCI ranking variables in both ecomplexity and py-ecomplexity to test.

# In[84]:


cdata.output['eci_rank'] = cdata.output.eci.rank(ascending=False)
cdata.output['pci_rank'] = cdata.output.pci.rank(ascending=False)


# In[85]:


cdata.output.describe()


# ### Compare against Stata output

# In[89]:


stata_output = pd.read_csv("data/processed/year_origin_hs92_4_ecomplexity_stata.csv")
stata_output.head()


# In[90]:


stata_output.describe()


# In[91]:


# Generate ranking vars for eci and pci
stata_output['eci_rank'] = stata_output.eci.rank(ascending=False)
stata_output['pci_rank'] = stata_output.pci.rank(ascending=False)


# In[92]:


# Check each var against stata output
def check_var(merged_outputs, varname):
    return(merged_outputs[~np.isclose(merged_outputs[varname+"_py"], merged_outputs[varname+"_st"],
                                      equal_nan=True)])


# In[93]:


def check_mismatch_stats(merged_outputs, varlist):
    res = {x:len(check_var(merged_outputs, x)) for x in varlist}
    return(res)


# In[94]:


merged_outputs = cdata.output.merge(stata_output, how='outer', left_on=['time','loc','prod'],
                                    right_on=['year','origin','hs92'], indicator=True,
                                    suffixes=['_py','_st'])


# In[95]:


merged_outputs._merge.value_counts()


# There seem to be some "left_only", that means that py-ecomplexity is generating some extra records

# In[96]:


wrong_var = check_var(merged_outputs, "eci")
wrong_var.head()


# In[97]:


check_mismatch_stats(merged_outputs, ['rca','diversity','ubiquity','eci','pci','eci_rank','pci_rank'])


# In[101]:


ax = sns.distplot(cdata.output.eci, hist=False)
ax = sns.distplot(stata_output.eci, hist=False, ax=ax)
plt.show()


# ### Testing Area
# Below this is temp testing area

# In[22]:


data.columns


# In[ ]:


data.desc


# In[ ]:


check.diversity[:,:,np.newaxis].shape


# In[ ]:


a = np.arange(24).reshape(2,3,4)
b = np.arange(12).reshape(3,4)[np.newaxis,:,:]
print(a)
print(b)


# In[ ]:


a.ravel()


# In[ ]:


b.repeat(a.shape[0], axis=0)


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

