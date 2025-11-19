"""Tests for the main ecomplexity module."""

import pandas as pd

from ecomplexity import ecomplexity


class TestEcomplexity:
    """Test suite for ecomplexity calculations."""

    def test_ecomplexity_basic_run(
        self, trade_data_fixture, trade_cols_mapping_fixture
    ):
        """Test that ecomplexity runs successfully with valid input."""
        result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that expected columns are present
        assert "eci" in result.columns
        assert "pci" in result.columns

        # Result may have more rows than input due to rectangularization
        # (creates full location×product matrix for each time period)
        assert len(result) >= len(trade_data_fixture)

    def test_ecomplexity_output_columns(
        self, trade_data_fixture, trade_cols_mapping_fixture
    ):
        """Test that ecomplexity returns expected output columns."""
        result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

        # Original columns should be preserved
        for col in trade_data_fixture.columns:
            assert col in result.columns

        # New complexity columns should be added
        expected_new_cols = ["eci", "pci", "diversity", "ubiquity", "rca", "mcp"]
        for col in expected_new_cols:
            assert col in result.columns

    def test_ecomplexity_no_null_eci_pci(
        self, trade_data_fixture, trade_cols_mapping_fixture
    ):
        """Test that ECI and PCI calculations don't produce all NaN values."""
        result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

        # At least some values should be non-null
        assert not result["eci"].isna().all()
        assert not result["pci"].isna().all()

    # TODO: Add more tests for:
    # - Different presence_test options (rca, rpop, manual)
    # - RCA and RPOP threshold variations
    # - Edge cases (single location, single product, etc.)
    # - Invalid input handling
    # - Log-supermodularity checking
    # - Verbose output
