"""Script to generate test fixtures from Stata ground truth output.

This script creates a small subset of the full Stata output (2 years, 10 countries, 30 products)
for use in fast-running tests that validate against Stata ground truth.
"""

from pathlib import Path

import pandas as pd

# Paths
ROOT = Path(__file__).parent.parent
STATA_OUTPUT = (
    ROOT / "data/processed/year_origin_hs92_4_ecomplexity_stata-1995-1996.csv"
)
RAW_DATA = ROOT / "data/raw/year_origin_hs92_4.tsv"
POP_DATA = ROOT / "data/raw/pop.csv"
FIXTURES_DIR = ROOT / "tests/fixtures"

# Parameters
N_COUNTRIES = 10
N_PRODUCTS = 30
YEARS = [1995, 1996]


def select_countries_and_products(stata_df):
    """Select countries and products with good coverage across both years.

    Prioritizes countries/products that:
    1. Appear in both years
    2. Have non-zero export values in at least one year
    3. Have reasonable diversity (not too few or too many products)
    """
    # Filter to years of interest
    stata_df = stata_df[stata_df["year"].isin(YEARS)].copy()

    # Find countries that appear in both years
    countries_by_year = stata_df.groupby("year")["origin"].unique()
    countries_both_years = set(countries_by_year[YEARS[0]]) & set(
        countries_by_year[YEARS[1]]
    )

    # Filter to countries in both years
    stata_df_filtered = stata_df[stata_df["origin"].isin(countries_both_years)].copy()

    # Calculate diversity (number of products with MCP=1) per country per year
    diversity = (
        stata_df_filtered[stata_df_filtered["M"] == 1]
        .groupby(["year", "origin"])["hs92"]
        .nunique()
        .reset_index(name="diversity")
    )

    # Get average diversity across years
    avg_diversity = diversity.groupby("origin")["diversity"].mean().reset_index()
    avg_diversity = avg_diversity.sort_values("diversity", ascending=False)

    # Select top N_COUNTRIES countries with good diversity
    # But also ensure we get a mix (not all high-diversity countries)
    selected_countries = avg_diversity.head(N_COUNTRIES * 2)["origin"].tolist()
    # Take every other one to get a mix
    selected_countries = selected_countries[::2][:N_COUNTRIES]

    # Now find products that appear with these countries
    stata_df_countries = stata_df_filtered[
        stata_df_filtered["origin"].isin(selected_countries)
    ]

    # Find products that appear in both years
    products_by_year = stata_df_countries.groupby("year")["hs92"].unique()
    products_both_years = set(products_by_year[YEARS[0]]) & set(
        products_by_year[YEARS[1]]
    )

    # Calculate ubiquity (number of countries with MCP=1) per product per year
    ubiquity = (
        stata_df_countries[stata_df_countries["M"] == 1]
        .groupby(["year", "hs92"])["origin"]
        .nunique()
        .reset_index(name="ubiquity")
    )

    # Get average ubiquity across years
    avg_ubiquity = ubiquity.groupby("hs92")["ubiquity"].mean().reset_index()
    avg_ubiquity = avg_ubiquity.sort_values("ubiquity", ascending=False)

    # Select products that appear in both years
    products_in_both = avg_ubiquity[avg_ubiquity["hs92"].isin(products_both_years)]

    # Select top N_PRODUCTS products
    selected_products = products_in_both.head(N_PRODUCTS)["hs92"].tolist()

    return selected_countries, selected_products


def generate_fixtures():
    """Generate fixture files from Stata output."""
    print("Loading Stata output...")
    stata_df = pd.read_csv(STATA_OUTPUT)

    print(f"Full dataset: {len(stata_df)} rows")
    print(f"Countries: {stata_df['origin'].nunique()}")
    print(f"Products: {stata_df['hs92'].nunique()}")

    # Select countries and products
    print(f"\nSelecting {N_COUNTRIES} countries and {N_PRODUCTS} products...")
    selected_countries, selected_products = select_countries_and_products(stata_df)

    print(f"Selected {len(selected_countries)} countries: {selected_countries[:5]}...")
    print(f"Selected {len(selected_products)} products: {selected_products[:5]}...")

    # Filter Stata output to selected subset
    stata_fixture = stata_df[
        (stata_df["origin"].isin(selected_countries))
        & (stata_df["hs92"].isin(selected_products))
        & (stata_df["year"].isin(YEARS))
    ].copy()

    print(f"\nFiltered Stata output: {len(stata_fixture)} rows")
    print(f"Countries in fixture: {stata_fixture['origin'].nunique()}")
    print(f"Products in fixture: {stata_fixture['hs92'].nunique()}")

    # Save Stata fixture
    FIXTURES_DIR.mkdir(exist_ok=True)
    stata_fixture_path = FIXTURES_DIR / "stata_ground_truth.csv"
    stata_fixture.to_csv(stata_fixture_path, index=False)
    print(f"\nSaved Stata fixture to: {stata_fixture_path}")

    # Now create corresponding input data fixture
    print("\nLoading raw data...")
    raw_data = pd.read_csv(RAW_DATA, sep="\t")

    # Filter raw data to match fixture
    input_fixture = raw_data[
        (raw_data["origin"].isin(selected_countries))
        & (raw_data["hs92"].isin(selected_products))
        & (raw_data["year"].isin(YEARS))
    ].copy()

    # Keep only export_val column (drop import columns)
    input_fixture = input_fixture[["year", "origin", "hs92", "export_val"]].copy()

    # Remove rows with null or zero export values (matching Stata preprocessing)
    input_fixture = input_fixture[
        input_fixture["export_val"].notna() & (input_fixture["export_val"] != 0)
    ].copy()

    print(f"Input fixture: {len(input_fixture)} rows")
    print(f"Countries in input: {input_fixture['origin'].nunique()}")
    print(f"Products in input: {input_fixture['hs92'].nunique()}")

    # Save input fixture
    input_fixture_path = FIXTURES_DIR / "trade_data_sample.csv"
    input_fixture.to_csv(input_fixture_path, index=False)
    print(f"\nSaved input fixture to: {input_fixture_path}")

    # Now create population data fixture
    print("\nLoading population data...")
    pop = pd.read_csv(POP_DATA)

    # Process population data similar to adhoc test
    # Drop metadata columns
    pop = pop.drop(columns=["Series Name", "Series Code", "Country Name"])

    # Extract year columns (format: "1995 [YR1995]")
    pop_yearcols = [x for x in pop.columns if x != "Country Code"]
    pop_yearcols_clean = [x[0:4] for x in pop_yearcols]
    pop.columns = ["cntry_code"] + pop_yearcols_clean

    # Convert year columns to numeric
    for x in pop_yearcols_clean:
        pop[x] = pd.to_numeric(pop[x], errors="coerce")

    # Filter to years of interest
    pop_yearcols_filtered = [str(y) for y in YEARS if str(y) in pop_yearcols_clean]

    # Only keep countries with data available for all years of interest
    num_years_available = pop[pop_yearcols_filtered].notnull().sum(axis=1)
    pop = pop[num_years_available == len(pop_yearcols_filtered)]

    # Reshape to long format
    pop_long = pop.melt(
        "cntry_code",
        value_vars=pop_yearcols_filtered,
        var_name="year",
        value_name="population",
    )
    pop_long["year"] = pd.to_numeric(pop_long["year"], errors="coerce").astype(int)
    pop_long = pop_long[["year", "cntry_code", "population"]]

    # Convert country codes to lowercase (matching trade data format)
    pop_long["cntry_code"] = pop_long["cntry_code"].str.lower()

    # Filter to selected countries and years
    pop_fixture = pop_long[
        (pop_long["cntry_code"].isin(selected_countries))
        & (pop_long["year"].isin(YEARS))
    ].copy()

    # Rename to match expected format (year, origin, population)
    # The ecomplexity function expects columns matching cols_input, so we'll use "origin"
    pop_fixture = pop_fixture.rename(columns={"cntry_code": "origin"})

    print(f"Population fixture: {len(pop_fixture)} rows")
    print(f"Countries in population: {pop_fixture['origin'].nunique()}")
    print(f"Years in population: {pop_fixture['year'].nunique()}")

    # Save population fixture
    pop_fixture_path = FIXTURES_DIR / "population_data_sample.csv"
    pop_fixture.to_csv(pop_fixture_path, index=False)
    print(f"\nSaved population fixture to: {pop_fixture_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Fixture generation complete!")
    print("=" * 60)
    print(f"Years: {YEARS}")
    print(f"Countries: {len(selected_countries)}")
    print(f"Products: {len(selected_products)}")
    print(f"Stata fixture rows: {len(stata_fixture)}")
    print(f"Input fixture rows: {len(input_fixture)}")
    print("\nFiles created:")
    print(f"  - {stata_fixture_path}")
    print(f"  - {input_fixture_path}")
    print(f"  - {pop_fixture_path}")


if __name__ == "__main__":
    generate_fixtures()
