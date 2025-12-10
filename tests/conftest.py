"""Pytest configuration and shared fixtures for ecomplexity tests."""

from pathlib import Path

import pandas as pd
import pytest

# ============================================================================
# Helper Functions
# ============================================================================


def load_fixture(filename: str) -> pd.DataFrame:
    """Load a fixture file from the fixtures directory.

    Args:
        filename: Name of the fixture file (e.g., 'trade_data_sample.csv')

    Returns:
        pd.DataFrame: Loaded fixture data

    Raises:
        pytest.skip: If fixture file not found
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    path = fixtures_dir / filename

    if not path.exists():
        pytest.skip(
            f"Fixture file not found: {path}. "
            "Run 'stata -b do generate_stata_fixtures.do' in the tests directory "
            "to generate Stata fixtures, or run 'python tests/generate_fixtures.py' "
            "for Python-generated fixtures."
        )

    return pd.read_csv(path)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def population_data_fixture():
    """Load real population data fixture for RPOP calculations.

    This fixture contains real population data (WDI) for the fixture countries and years.
    Use this for tests that need population data for RPOP calculations.

    Returns:
        pd.DataFrame: Population data with columns: year, origin, population
    """
    return load_fixture("population_data_sample.csv")


@pytest.fixture
def trade_data_fixture():
    """Load trade data fixture for tests.

    This fixture contains a subset of real trade data (2 years, 10 countries, 30 products)
    from the Atlas of Economic Complexity (sourced from UN COMTRADE). Use this for all tests
    that need trade data.

    Returns:
        pd.DataFrame: Trade data with columns: year, origin, hs92, export_val
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    input_path = fixtures_dir / "trade_data_sample.csv"

    if not input_path.exists():
        pytest.skip(
            f"Fixture file not found: {input_path}. "
            "Run 'python tests/generate_fixtures.py' to generate fixtures."
        )

    return pd.read_csv(input_path)


@pytest.fixture
def trade_cols_mapping_fixture():
    """Column mapping for trade data fixture.

    Returns:
        dict: Mapping of internal names to column names for fixture data
    """
    return {
        "time": "year",
        "loc": "origin",
        "prod": "hs92",
        "val": "export_val",
    }


@pytest.fixture
def stata_ground_truth_fixture():
    """Load Stata ground truth output for comparison tests.

    This fixture contains Stata ecomplexity output for the fixture data subset.
    Use this to validate that Python implementation matches Stata reference.

    Returns:
        pd.DataFrame: Stata output with columns: year, origin, hs92, export_val,
            rca, M, density, eci, pci, diversity, ubiquity, coi, cog
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    stata_path = fixtures_dir / "stata_ground_truth.csv"

    if not stata_path.exists():
        pytest.skip(
            f"Fixture file not found: {stata_path}. "
            "Run 'python tests/generate_fixtures.py' to generate fixtures."
        )

    return pd.read_csv(stata_path)
