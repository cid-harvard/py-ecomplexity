"""Test that Stata fixture fixtures load correctly and can be used for comparison."""

import pandas as pd

from ecomplexity import ecomplexity


def test_stata_fixtures_load(
    trade_data_fixture, trade_cols_mapping_fixture, stata_ground_truth_fixture
):
    """Test that all fixtures load correctly."""
    # Check trade data fixture
    assert isinstance(trade_data_fixture, pd.DataFrame)
    assert len(trade_data_fixture) > 0
    assert "year" in trade_data_fixture.columns
    assert "origin" in trade_data_fixture.columns
    assert "hs92" in trade_data_fixture.columns
    assert "export_val" in trade_data_fixture.columns

    # Check column mapping
    assert isinstance(trade_cols_mapping_fixture, dict)
    assert all(
        key in trade_cols_mapping_fixture for key in ["time", "loc", "prod", "val"]
    )

    # Check Stata ground truth fixture
    assert isinstance(stata_ground_truth_fixture, pd.DataFrame)
    assert len(stata_ground_truth_fixture) > 0
    expected_cols = [
        "year",
        "origin",
        "hs92",
        "eci",
        "pci",
        "rca",
        "M",
        "diversity",
        "ubiquity",
    ]
    for col in expected_cols:
        assert col in stata_ground_truth_fixture.columns

    # Check that years match
    assert set(trade_data_fixture["year"].unique()) == set(
        stata_ground_truth_fixture["year"].unique()
    )

    # Check that countries match
    fixture_countries = set(trade_data_fixture["origin"].unique())
    stata_countries = set(stata_ground_truth_fixture["origin"].unique())
    assert fixture_countries == stata_countries

    # Check that products match
    fixture_products = set(trade_data_fixture["hs92"].unique())
    stata_products = set(stata_ground_truth_fixture["hs92"].unique())
    assert fixture_products == stata_products


def test_stata_fixture_basic_calculation(
    trade_data_fixture, trade_cols_mapping_fixture
):
    """Test that ecomplexity can run on fixture data."""
    result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that expected columns are present
    assert "eci" in result.columns
    assert "pci" in result.columns
    assert "rca" in result.columns
    assert "mcp" in result.columns
    assert "diversity" in result.columns
    assert "ubiquity" in result.columns

    # Check that we have reasonable number of rows
    assert len(result) > 0

    # Check that ECI and PCI are not all NaN
    assert not result["eci"].isna().all()
    assert not result["pci"].isna().all()


def test_stata_fixture_comparison_structure(
    trade_data_fixture, trade_cols_mapping_fixture, stata_ground_truth_fixture
):
    """Test that fixture data can be merged with Stata output for comparison."""
    # Run Python calculation
    py_result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

    # Merge with Stata output
    merged = py_result.merge(
        stata_ground_truth_fixture,
        on=["year", "origin", "hs92"],
        suffixes=("_py", "_st"),
        how="outer",
        indicator=True,
    )

    # Check that merge worked
    assert len(merged) > 0

    # Check that we have both Python and Stata columns
    assert "eci_py" in merged.columns
    assert "eci_st" in merged.columns
    assert "pci_py" in merged.columns
    assert "pci_st" in merged.columns

    # Check that most rows match (some may differ due to Stata preprocessing)
    # We expect at least some matches
    both_present = merged[merged["eci_py"].notna() & merged["eci_st"].notna()]
    assert len(both_present) > 0
