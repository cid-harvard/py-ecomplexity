# Economic Complexity and Product Complexity

Python package to calculate economic complexity indices.

STATA implementation of the economic complexity index available at: https://github.com/cid-harvard/ecomplexity

Explore complexity and associated data using Harvard CID's Atlas tool: http://atlas.cid.harvard.edu

### Tutorial

**Installation**:
At terminal: `pip install git+https://github.com/cid-harvard/py-ecomplexity@master`

**Usage**:
```python
from ecomplexity import ecomplexity
from ecomplexity import proximity

# Import trade data from CID Atlas
data_url = "https://intl-atlas-downloads.s3.amazonaws.com/country_hsproduct2digit_year.csv.zip"
data = pd.read_csv(data_url, compression="zip", low_memory=False)
data = data[['year','location_code','hs_product_code','export_value']]

# Calculate complexity
trade_cols = {'time':'year', 'loc':'location_code', 'prod':'hs_product_code', 'val':'export_value'}
cdata = ecomplexity(data, trade_cols)

# Calculate proximity matrix
prox_df = proximity(data, trade_cols)
```

### TODO:

- There are very minor differences in the values of density, COI and COG between STATA and Python due to the way matrix computations are handled by the two. These should be aligned in the future.
- knn options for density: in the future, allow knn parameter for density calculation
