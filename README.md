# Economic Complexity and Product Complexity

Python package to calculate economic complexity indices.

STATA implementation of the economic complexity index available at: https://github.com/cid-harvard/ecomplexity

Explore complexity and associated data using Harvard CID's Atlas tool: http://atlas.cid.harvard.edu

#### TODO:

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
