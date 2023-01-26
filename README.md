# Economic Complexity and Product Complexity

By the Growth Lab at Harvard's Center for International Development

This package is part of Harvard Growth Lab’s portfolio of software packages, digital products and interactive data visualizations. To browse our entire portfolio, please visit [growthlab.app](growthlab.app). To learn more about our research, please visit [Harvard Growth Lab’s home page](https://growthlab.cid.harvard.edu/).

# About
Python package to calculate economic complexity indices.

STATA implementation of the economic complexity index available at: <https://github.com/cid-harvard/ecomplexity>

Explore complexity and associated data using Harvard CID's Atlas tool: <http://atlas.cid.harvard.edu>

## Tutorial

**Installation**:
At terminal: `pip install ecomplexity`

If you wish to install the latest version of the package under development, you can install directly from GitHub:
`pip install git+https://github.com/cid-harvard/py-ecomplexity@develop`

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

**Arguments**:

```text
data: pandas dataframe containing production / trade data.
    Including variables indicating time, location, product and value
cols_input: dict of column names for time, location, product and value.
    Example: {'time':'year', 'loc':'origin', 'prod':'hs92', 'val':'export_val'}
presence_test: str for test used for presence of industry in location.
    One of "rca" (default), "rpop", "both", or "manual".
    Determines which values are used for M_cp calculations.
    If "manual", M_cp is taken as given from the "value" column in data
val_errors_flag: {'coerce','ignore','raise'}. Passed to pd.to_numeric
    *default* coerce.
rca_mcp_threshold: numeric indicating RCA threshold beyond which mcp is 1.
    *default* 1.
rpop_mcp_threshold: numeric indicating RPOP threshold beyond which mcp is 1.
    *default* 1. Only used if presence_test is not "rca".
pop: pandas df, with time, location and corresponding population, in that order.
    Not required if presence_test is "rca" (default).
continuous: Used to calculate product proximities, indicates whether
    to consider correlation of every product pair (True) or product
    co-occurrence (False). *default* False.
asymmetric: Used to calculate product proximities, indicates whether
    to generate asymmetric proximity matrix (True) or symmetric (False).
    *default* False.
knn: Number of nearest neighbors from proximity matrix to use to calculate
    density. Will use entire proximity matrix if None.
    *default* None.
```

## FAQ

- Why are ECI and PCI are both normalized using ECI's mean and std. dev?
    + This normalization preserves the property that ECI = (mean of PCI of products for which MCP=1)


### References

- Hausmann, R., Hidalgo, C. A., Bustos, S., Coscia, M., Simoes, A., & Yıldırım, M. (2013). The Atlas of Economic Complexity: Mapping Paths to Prosperity (Part 1). Retrieved from <https://growthlab.cid.harvard.edu/files/growthlab/files/atlas_2013_part1.pdf>
- Hidalgo, C. A., Klinger, B., Barabasi, A.-L., & Hausmann, R. (2007). The Product Space Conditions the Development of Nations. Science, 317(5837), 482–487. <http://doi.org/10.1126/science.1144581>
