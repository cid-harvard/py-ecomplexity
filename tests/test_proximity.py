"""Tests for proximity calculations."""

import pandas as pd

from ecomplexity import proximity


class TestProximity:
    """Test suite for product proximity calculations."""

    def test_proximity_basic_run(self, trade_data_fixture, trade_cols_mapping_fixture):
        """Test that proximity calculation runs successfully."""
        result = proximity(trade_data_fixture, trade_cols_mapping_fixture)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that proximity column is present
        assert "proximity" in result.columns

        # Check that result has product pair columns (actual names depend on input)
        assert len(result.columns) == 4  # year, prod1, prod2, proximity

    def test_proximity_output_format(
        self, trade_data_fixture, trade_cols_mapping_fixture
    ):
        """Test that proximity returns edgelist format."""
        result = proximity(trade_data_fixture, trade_cols_mapping_fixture)

        # Check that proximity values are between 0 and 1
        assert (result["proximity"] >= 0).all()
        assert (result["proximity"] <= 1).all()

        # Check that we have product pairs (columns 1 and 2 are the product codes)
        prod_col1 = result.columns[1]
        prod_col2 = result.columns[2]
        assert result[prod_col1].notna().all()
        assert result[prod_col2].notna().all()

    def test_proximity_discrete_vs_continuous(
        self, trade_data_fixture, trade_cols_mapping_fixture
    ):
        """Test both discrete and continuous proximity calculations."""
        discrete = proximity(
            trade_data_fixture, trade_cols_mapping_fixture, continuous=False
        )
        continuous = proximity(
            trade_data_fixture, trade_cols_mapping_fixture, continuous=True
        )

        # Both should return DataFrames
        assert isinstance(discrete, pd.DataFrame)
        assert isinstance(continuous, pd.DataFrame)

        # Both should have the same structure
        assert list(discrete.columns) == list(continuous.columns)

    # TODO: Add more tests for:
    # - Symmetric vs asymmetric proximity
    # - Diagonal values (self-proximity)
    # - Edge cases (single product, no trade, etc.)
    # - Proximity values make sense (e.g., self-proximity = 1 for discrete)
    # - Different time periods handled correctly
