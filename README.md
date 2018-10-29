# Economic Complexity and Product Complexity

THIS PACKAGE IS A WORK IN PROGRESS, DO NOT USE FOR PRODUCTION

Python package to calculate economic complexity indices.

STATA implementation of the economic complexity index available at: https://github.com/cid-harvard/ecomplexity

Explore complexity and associated data using Harvard CID's Atlas tool: http://atlas.cid.harvard.edu

The aim is to replicate the STATA ecomplexity package's features:
- Args:
    + dict: colnames for time,loc,prod,val
    + rca_mcp_threshold: mcp cutoff
    + rpop
    + pop
    + cont
    + asym
    + knn

- Returns:
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


#### Notes

Currently, this handles NaN's by coercing them to zero. This is true for both NaN's in the trade / production values, and those in the population values (for rpop)

For our test dataset containing world trade, the STATA ecomplexity package takes around 4.88 mins, and the py-ecomplexity package takes around 40 seconds. This will be even faster if parallelized.

#### TODO:

- Make the data outputs conform to stata output format. Currently ndarrays, convert to pandas df's with time, location and products explicitly listed.
- Test the code against stata output
- Parallellize the numpy vectorization. Currently runs on a single thread on a single core.
