# Economic Complexity and Product Complexity

NOTE: Density calculations not yet tested! ECI, PCI, diversity and ubiquity tested.

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

- Test density, make it conform to STATA package outputs
- Develop COI, COG calculations
- knn options for density: in the future, allow knn parameter for density calculation

The aim is to replicate the STATA ecomplexity package's features:
- Args:
    + dict: colnames for time,loc,prod,val
    + rca_mcp_threshold: mcp cutoff
    + rpop
    + pop
    + cont
    + asym
    + knn

- Returns: pandas df with original data and following additional cols:
    + eci
    + pci
    + density
    + coi
    + cog
    + diversity
    + ubiquity
    + rca
    + rpop
    + M
