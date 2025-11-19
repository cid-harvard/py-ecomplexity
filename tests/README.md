# Tests

This directory contains the test suite for py-ecomplexity.

## Structure

- `conftest.py` - Shared pytest fixtures for sample data
- `test_ecomplexity.py` - Tests for main ecomplexity calculations
- `test_proximity.py` - Tests for proximity calculations
- `test_density.py` - Tests for density calculations

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=ecomplexity

# Run specific test file
uv run pytest tests/test_ecomplexity.py

# Run specific test
uv run pytest tests/test_ecomplexity.py::TestEcomplexity::test_ecomplexity_basic_run
```

## Writing Tests

### Test Organization

- Group related tests in classes (e.g., `TestEcomplexity`)
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Focus on integration tests over unit tests
- Test both happy paths and edge cases

### Using Fixtures

The `conftest.py` file provides shared fixtures:

- `trade_data_fixture` - Real trade data subset (10 countries, 30 products, 2 years) from Atlas of Economic Complexity (UN COMTRADE)
- `trade_cols_mapping_fixture` - Column mapping for trade data fixture (maps to columns: year, origin, hs92, export_val)
- `stata_ground_truth_fixture` - Stata ecomplexity output for fixture data (ground truth for validation)
- `population_data_fixture` - Real population data (WDI) for fixture countries and years, used for RPOP tests

Example:

```python
def test_my_feature(trade_data_fixture, trade_cols_mapping_fixture):
    result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)
    assert "eci" in result.columns
```

For Stata ground truth validation:

```python
def test_matches_stata(trade_data_fixture, trade_cols_mapping_fixture, stata_ground_truth_fixture):
    py_result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)
    merged = py_result.merge(stata_ground_truth_fixture, on=["year", "origin", "hs92"],
                             suffixes=("_py", "_st"))
    assert np.allclose(merged["eci_py"], merged["eci_st"], rtol=1e-5, equal_nan=True)
```

See `tests/fixtures/README.md` for more details on fixtures.

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_dataset():
    # Test that takes a long time
    pass

@pytest.mark.integration
def test_full_pipeline():
    # Integration test
    pass
```

Run specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

## Adding New Tests

When adding new functionality:

1. Add tests in the appropriate test file (or create a new one)
2. Use fixtures from `conftest.py` or create new ones
3. Test both expected behavior and edge cases
4. Ensure tests run quickly (< 1 second each)
5. Document what the test is verifying in the docstring

## Notes

- Tests use a fixed random seed (42) for reproducibility
- Sample data is small to keep tests fast
- RuntimeWarnings in density/COI calculations are expected with small sample data
