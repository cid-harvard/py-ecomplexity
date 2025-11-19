"""Tests for density calculations."""

from ecomplexity import ecomplexity


class TestDensity:
    """Test suite for density calculations."""

    def test_density_in_output(self, trade_data_fixture, trade_cols_mapping_fixture):
        """Test that density is calculated and included in output."""
        result = ecomplexity(trade_data_fixture, trade_cols_mapping_fixture)

        # Check that density column exists
        assert "density" in result.columns

        # Check that density values are between 0 and 1
        density_values = result["density"].dropna()
        if len(density_values) > 0:
            assert (density_values >= 0).all()
            assert (density_values <= 1).all()

    # TODO: Add more tests for:
    # - KNN density (with scikit-learn installed)
    # - Standard density vs KNN density comparison
    # - Custom proximity matrix input
    # - Edge cases (no neighbors, etc.)
