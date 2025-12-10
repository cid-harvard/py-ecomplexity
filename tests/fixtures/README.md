# Test Fixtures

Test fixtures for validating the Python implementation against Stata ground truth.

## Overview

This directory contains:
1. **Input fixtures** - Sample trade and population data
2. **Stata ground truth fixtures** - Output from Stata ecomplexity for various parameter combinations

## Input Files

| File | Description |
|------|-------------|
| `trade_data_sample.csv` | Real trade data subset (2 years: 1995-1996, 10 countries, 30 products) |
| `population_data_sample.csv` | Population data (WDI) for fixture countries and years |
| `manual_mcp_sample.csv` | Pre-computed binary MCP matrix for manual MCP tests |

## Stata Ground Truth Files

| File | Description | Python Parameters | Stata Command |
|------|-------------|-------------------|---------------|
| `stata_ground_truth.csv` | Default (RCA=1, discrete, symmetric) | default | `ecomplexity export_val, i(origin) p(hs92) t(year)` |
| `stata_rca_05.csv` | RCA threshold 0.5 | `rca_mcp_threshold=0.5` | `ecomplexity export_val, ... rca(0.5)` |
| `stata_rca_2.csv` | RCA threshold 2.0 | `rca_mcp_threshold=2.0` | `ecomplexity export_val, ... rca(2)` |
| `stata_rpop_default.csv` | RPOP threshold 1.0 | `presence_test="rpop"` | `ecomplexity export_val, ... pop(population) rpop(1)` |
| `stata_rpop_2.csv` | RPOP threshold 2.0 | `presence_test="rpop", rpop_mcp_threshold=2.0` | `ecomplexity export_val, ... pop(population) rpop(2)` |
| `stata_discrete_asymmetric.csv` | Asymmetric proximity | `asymmetric=True` | `ecomplexity export_val, ... asym` |
| `stata_continuous.csv` | Continuous (Pearson) proximity | `continuous=True` | `ecomplexity export_val, ... cont` |
| `stata_continuous_rpop.csv` | Continuous + RPOP | `continuous=True, presence_test="rpop"` | `ecomplexity export_val, ... pop(population) rpop(1) cont` |
| `stata_knn_5.csv` | KNN density (k=5) | `knn=5` | `ecomplexity export_val, ... knn(5)` |
| `stata_knn_10.csv` | KNN density (k=10) | `knn=10` | `ecomplexity export_val, ... knn(10)` |
| `stata_knn_20.csv` | KNN density (k=20) | `knn=20` | `ecomplexity export_val, ... knn(20)` |
| `stata_manual_mcp.csv` | Manual MCP input | `presence_test="manual"` | `ecomplexity mcp, ... bi` |
| `stata_rca_rpop_combined.csv` | Combined RCA + RPOP (union) | N/A (proximity.py only) | `ecomplexity export_val, ... pop(population) rca(1) rpop(1)` |
| `stata_continuous_knn_10.csv` | Continuous + KNN=10 | `continuous=True, knn=10` | `ecomplexity export_val, ... cont knn(10)` |
| `stata_single_year_1995.csv` | Single year (1995) | filter to single year | Filter to year=1995 |

### Proximity Matrix Files (genproximity command)

| File | Description | Python Parameters |
|------|-------------|-------------------|
| `stata_proximity_discrete_sym.csv` | Proximity matrix (discrete, symmetric) | `continuous=False, asymmetric=False` |
| `stata_proximity_discrete_asym.csv` | Proximity matrix (discrete, asymmetric) | `continuous=False, asymmetric=True` |
| `stata_proximity_continuous.csv` | Proximity matrix (continuous/Pearson) | `continuous=True` |

## Generating Fixtures

### Step 1: Install Stata ecomplexity package

```stata
net install ecomplexity, from("https://raw.githubusercontent.com/cid-harvard/ecomplexity/master/") force
net install genproximity, from("https://raw.githubusercontent.com/cid-harvard/ecomplexity/master/") force
```

### Step 2: Generate fixtures

```bash
cd tests
stata -b do generate_stata_fixtures.do
```

This will create all the fixture files listed above.

## Using Fixtures in Tests

Fixtures are loaded via pytest fixtures in `tests/conftest.py`:

```python
def test_matches_stata(trade_data_fixture, trade_cols_mapping_fixture, stata_ground_truth_fixture):
    py_result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)
    merged = py_result.merge(
        stata_ground_truth_fixture,
        on=["year", "origin", "hs92"],
        suffixes=("_py", "_st")
    )
    np.testing.assert_allclose(
        merged["eci_py"],
        merged["eci_st"],
        rtol=1e-5,
        equal_nan=True
    )
```

## Column Mapping

| Python Column | Stata Column | Description |
|---------------|--------------|-------------|
| `mcp` | `M` | Binary presence matrix |
| `eci` | `eci` | Economic Complexity Index |
| `pci` | `pci` | Product Complexity Index |
| `rca` | `rca` | Revealed Comparative Advantage |
| `rpop` | `rpop` | Revealed Population |
| `density` | `density` | Product density |
| `diversity` | `diversity` | Number of products per location |
| `ubiquity` | `ubiquity` | Number of locations per product |
| `coi` | `coi` | Complexity Outlook Index |
| `cog` | `cog` | Complexity Outlook Gain |

## Tolerance Guidelines

| Column Type | Tolerance | Reason |
|-------------|-----------|--------|
| Binary (mcp) | Exact | Should be identical |
| Integer (diversity, ubiquity) | Exact | Should be identical |
| RCA, RPOP | rtol=1e-5 | Minor floating point differences |
| ECI, PCI | rtol=1e-5 | Normalized values |
| density, coi, cog | rtol=1e-4 | Derived metrics, more tolerance |

## Data Characteristics

- **Countries** (10): aut, blx, che, deu, dnk, esp, gbr, ita, pol, svn
- **Products** (30): HS92 4-digit codes
- **Years** (2): 1995, 1996
- **Total rows**: ~600 (10 countries × 30 products × 2 years)
