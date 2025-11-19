# Test Fixtures

Test fixtures for validating the Python implementation against Stata ground truth.

## Files

- `trade_data_sample.csv`: Real trade data subset (2 years: 1995-1996, 10 countries, 30 products)
- `stata_ground_truth.csv`: Stata ecomplexity output for the trade data subset
- `population_data_sample.csv`: Real population data (WDI) for fixture countries and years

## Generating Fixtures

```bash
uv run python tests/generate_fixtures.py
```

## Using Fixtures

Fixtures are loaded via pytest fixtures in `tests/conftest.py`:

- `trade_data_fixture`: Trade data DataFrame (from Atlas of Economic Complexity / UN COMTRADE)
- `trade_cols_mapping_fixture`: Column mapping dictionary for fixture data
- `stata_ground_truth_fixture`: Stata ecomplexity output DataFrame for validation
- `population_data_fixture`: Population data DataFrame (from World Bank WDI) for RPOP calculations

Example:

```python
def test_matches_stata(trade_data_fixture, trade_cols_mapping_fixture, stata_ground_truth_fixture):
    py_result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)
    merged = py_result.merge(stata_ground_truth_fixture, on=["year", "origin", "hs92"],
                             suffixes=("_py", "_st"))
    assert np.allclose(merged["eci_py"], merged["eci_st"], rtol=1e-5, equal_nan=True)
```
