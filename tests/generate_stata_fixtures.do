/*******************************************************************************
 * generate_stata_fixtures.do
 *
 * This Stata do file generates ground truth fixtures for validating
 * py-ecomplexity against the Stata ecomplexity package.
 *
 * Usage:
 *   1. Install the Stata ecomplexity package:
 *      net install ecomplexity, from("https://raw.githubusercontent.com/cid-harvard/ecomplexity/master/") force
 *      net install genproximity, from("https://raw.githubusercontent.com/cid-harvard/ecomplexity/master/") force
 *
 *   2. Run this do file:
 *      cd /n/holystore01/LABS/hausmann_lab/users/shreyasgm/ecomplexity/tests
 *      do generate_stata_fixtures.do
 *
 *
 *******************************************************************************
 * FIXTURE OUTPUT FILES AND PARAMETER COMBINATIONS
 *******************************************************************************
 *
 * The table below shows all fixture files generated, the parameters being
 * tested, and the corresponding Stata command.
 *
 * Python Parameters (ecomplexity.py):
 *   - presence_test: "rca", "rpop", "manual"
 *   - rca_mcp_threshold: float (default 1.0)
 *   - rpop_mcp_threshold: float (default 1.0)
 *   - pop: DataFrame with population data
 *   - continuous: bool (default False)
 *   - asymmetric: bool (default False)
 *   - knn: int or None (default None)
 *   - proximity_edgelist: DataFrame (not testable against Stata)
 *
 * Python Parameters (proximity.py):
 *   - presence_test: "rca", "rpop", "both", "manual"
 *   - All other params same as ecomplexity.py
 *
 * Stata Parameters (ecomplexity/genproximity):
 *   - rca(#): RCA threshold
 *   - rpop(#): RPOP threshold
 *   - pop(var): Population variable
 *   - cont: Continuous proximity (Pearson correlation)
 *   - asym: Asymmetric proximity
 *   - knn(#): K-nearest neighbors for density
 *   - Binary input: auto-detected when varlist is 0/1
 *
 *------------------------------------------------------------------------------
 * ECOMPLEXITY FIXTURES
 *------------------------------------------------------------------------------
 * File Name                           | Parameters Tested                    | Stata Command
 * ------------------------------------|--------------------------------------|--------------------------------------------------
 * stata_ground_truth.csv              | rca_mcp_threshold=1, discrete, sym   | ecomplexity export_val, i(origin) p(hs92) t(year)
 * stata_rca_05.csv                    | rca_mcp_threshold=0.5                | ecomplexity export_val, ... rca(0.5)
 * stata_rca_2.csv                     | rca_mcp_threshold=2.0                | ecomplexity export_val, ... rca(2)
 * stata_rpop_default.csv              | presence_test="rpop", threshold=1    | ecomplexity export_val, ... pop(population) rpop(1)
 * stata_rpop_2.csv                    | presence_test="rpop", threshold=2    | ecomplexity export_val, ... pop(population) rpop(2)
 * stata_discrete_asymmetric.csv       | asymmetric=True                      | ecomplexity export_val, ... asym
 * stata_continuous.csv                | continuous=True                      | ecomplexity export_val, ... cont
 * stata_knn_5.csv                     | knn=5                                | ecomplexity export_val, ... knn(5)
 * stata_knn_10.csv                    | knn=10                               | ecomplexity export_val, ... knn(10)
 * stata_knn_20.csv                    | knn=20                               | ecomplexity export_val, ... knn(20)
 * stata_manual_mcp.csv                | presence_test="manual"               | ecomplexity M, ... bi
 * stata_rca_rpop_combined.csv         | presence_test="both" (RCA OR RPOP)   | ecomplexity export_val, ... pop(population) rca(1) rpop(1)
 * stata_continuous_knn_10.csv         | continuous=True, knn=10              | ecomplexity export_val, ... cont knn(10)
 * stata_single_year_1995.csv          | Single time period                   | (filter year==1995) ecomplexity export_val, ...
 * stata_continuous_rpop.csv           | continuous=True, presence_test="rpop"| ecomplexity export_val, ... pop(population) rpop(1) cont
 *
 *------------------------------------------------------------------------------
 * PROXIMITY FIXTURES (genproximity)
 *------------------------------------------------------------------------------
 * File Name                           | Parameters Tested                    | Stata Command
 * ------------------------------------|--------------------------------------|--------------------------------------------------
 * stata_proximity_discrete_sym.csv    | continuous=False, asymmetric=False   | genproximity export_val, i(origin) p(hs92) t(year)
 * stata_proximity_discrete_asym.csv   | continuous=False, asymmetric=True    | genproximity export_val, ... asym
 * stata_proximity_continuous.csv      | continuous=True                      | genproximity export_val, ... cont
 *
 *------------------------------------------------------------------------------
 * INPUT/SUPPORT FILES
 *------------------------------------------------------------------------------
 * File Name                           | Description
 * ------------------------------------|--------------------------------------
 * trade_data_sample.csv               | Input trade data (created by generate_fixtures.py)
 * population_data_sample.csv          | Input population data (created by generate_fixtures.py)
 * manual_mcp_sample.csv               | Pre-computed MCP matrix for manual MCP tests
 *
 *------------------------------------------------------------------------------
 * TOTAL: 18 fixture files (15 ecomplexity + 3 proximity)
 *------------------------------------------------------------------------------
 *
 ******************************************************************************/

clear all
set more off

* Set the working directory to the ecomplexity project root
cd "/n/holystore01/LABS/hausmann_lab/users/shreyasgm/ecomplexity"

* Define paths
local ecomplexity_dir = "/n/holystore01/LABS/hausmann_lab/users/shreyasgm/ecomplexity"
local tests_dir = "`ecomplexity_dir'/tests"
local fixtures_dir = "`tests_dir'/fixtures"
local input_file = "`fixtures_dir'/trade_data_sample.csv"
local pop_file = "`fixtures_dir'/population_data_sample.csv"

* Display progress header
di ""
di "=============================================="
di "Generating Stata Fixtures for py-ecomplexity"
di "=============================================="
di ""


/*******************************************************************************
 * STEP 1: Load trade data and verify
 ******************************************************************************/

di "Step 1: Loading trade data..."
import delimited "`input_file'", clear

* Check data loaded correctly
describe
summarize

di "Trade data loaded: `c(N)' observations"
di ""


/*******************************************************************************
 * STEP 2: Load population data (pre-created by generate_fixtures.py)
 ******************************************************************************/

di "Step 2: Loading population data..."

* Load population data from CSV
preserve
    import delimited "`pop_file'", clear
    describe
    list in 1/5
    tempfile popdata
    save `popdata'
restore

di "Population data loaded successfully"
di ""


/*******************************************************************************
 * STEP 3: Generate default RCA=1 fixture (baseline)
 ******************************************************************************/

di ""
di "Step 3: Generating stata_ground_truth.csv (default RCA=1, discrete, symmetric)..."

import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year)
export delimited "`fixtures_dir'/stata_ground_truth.csv", replace


/*******************************************************************************
 * STEP 4: Generate RCA threshold variations
 ******************************************************************************/

di ""
di "Step 4: Generating RCA threshold variations..."

* RCA = 0.5
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) rca(0.5)
export delimited "`fixtures_dir'/stata_rca_05.csv", replace
di "  Saved: stata_rca_05.csv"

* RCA = 2.0
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) rca(2)
export delimited "`fixtures_dir'/stata_rca_2.csv", replace
di "  Saved: stata_rca_2.csv"


/*******************************************************************************
 * STEP 5: Generate RPOP fixtures
 ******************************************************************************/

di ""
di "Step 5: Generating RPOP fixtures..."

* RPOP = 1 (default)
import delimited "`input_file'", clear
merge m:1 origin year using `popdata', nogen
ecomplexity export_val, i(origin) p(hs92) t(year) pop(population) rpop(1)
export delimited "`fixtures_dir'/stata_rpop_default.csv", replace
di "  Saved: stata_rpop_default.csv"

* RPOP = 2
import delimited "`input_file'", clear
merge m:1 origin year using `popdata', nogen
ecomplexity export_val, i(origin) p(hs92) t(year) pop(population) rpop(2)
export delimited "`fixtures_dir'/stata_rpop_2.csv", replace
di "  Saved: stata_rpop_2.csv"


/*******************************************************************************
 * STEP 6: Generate proximity method variations
 ******************************************************************************/

di ""
di "Step 6: Generating proximity method variations..."

* Discrete asymmetric
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) asym
export delimited "`fixtures_dir'/stata_discrete_asymmetric.csv", replace
di "  Saved: stata_discrete_asymmetric.csv"

* Continuous (Pearson correlation)
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) cont
export delimited "`fixtures_dir'/stata_continuous.csv", replace
di "  Saved: stata_continuous.csv"


/*******************************************************************************
 * STEP 7: Generate KNN density variations
 ******************************************************************************/

di ""
di "Step 7: Generating KNN density variations..."

* KNN = 5
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) knn(5)
export delimited "`fixtures_dir'/stata_knn_5.csv", replace
di "  Saved: stata_knn_5.csv"

* KNN = 10
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) knn(10)
export delimited "`fixtures_dir'/stata_knn_10.csv", replace
di "  Saved: stata_knn_10.csv"

* KNN = 20
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) knn(20)
export delimited "`fixtures_dir'/stata_knn_20.csv", replace
di "  Saved: stata_knn_20.csv"


/*******************************************************************************
 * STEP 8: Generate manual MCP fixture
 ******************************************************************************/

di ""
di "Step 8: Generating manual MCP fixture..."

* First, create MCP from default RCA>=1 run and save it
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year)

* Save the M column for manual input test
preserve
    keep year origin hs92 M
    * Export with "mcp" column name for Python test fixtures
    rename M mcp
    export delimited "`fixtures_dir'/manual_mcp_sample.csv", replace
    di "  Saved: manual_mcp_sample.csv"
    * Rename to a name that won't conflict with ecomplexity output variables
    rename mcp binary_presence
    tempfile mcp_data
    save `mcp_data'
restore

* Now test manual MCP input - reload original data and merge with MCP
import delimited "`input_file'", clear
merge 1:1 year origin hs92 using `mcp_data', nogen

* Use binary_presence as the input variable (named to avoid conflict with ecomplexity outputs)
* The bi option tells ecomplexity to treat this as a pre-computed binary presence matrix
ecomplexity binary_presence, i(origin) p(hs92) t(year) bi
export delimited "`fixtures_dir'/stata_manual_mcp.csv", replace
di "  Saved: stata_manual_mcp.csv"


/*******************************************************************************
 * STEP 9: Generate combined parameter fixtures
 ******************************************************************************/

di ""
di "Step 9: Generating combined parameter fixtures..."

* Combined RCA + RPOP (union of both criteria - presence_test="both" in Python proximity.py)
* Note: In Stata, specifying both rca() and rpop() uses union: MCP=1 if EITHER threshold is met
import delimited "`input_file'", clear
merge m:1 origin year using `popdata', nogen
ecomplexity export_val, i(origin) p(hs92) t(year) pop(population) rca(1) rpop(1)
export delimited "`fixtures_dir'/stata_rca_rpop_combined.csv", replace
di "  Saved: stata_rca_rpop_combined.csv"

* Continuous + KNN = 10
import delimited "`input_file'", clear
ecomplexity export_val, i(origin) p(hs92) t(year) cont knn(10)
export delimited "`fixtures_dir'/stata_continuous_knn_10.csv", replace
di "  Saved: stata_continuous_knn_10.csv"

* Continuous + RPOP (for testing continuous=True with presence_test="rpop")
import delimited "`input_file'", clear
merge m:1 origin year using `popdata', nogen
ecomplexity export_val, i(origin) p(hs92) t(year) pop(population) rpop(1) cont
export delimited "`fixtures_dir'/stata_continuous_rpop.csv", replace
di "  Saved: stata_continuous_rpop.csv"


/*******************************************************************************
 * STEP 10: Generate proximity matrix fixtures (genproximity command)
 ******************************************************************************/

di ""
di "Step 10: Generating proximity matrix fixtures..."

* Discrete symmetric proximity
import delimited "`input_file'", clear
genproximity export_val, i(origin) p(hs92) t(year)
export delimited "`fixtures_dir'/stata_proximity_discrete_sym.csv", replace
di "  Saved: stata_proximity_discrete_sym.csv"

* Discrete asymmetric proximity
import delimited "`input_file'", clear
genproximity export_val, i(origin) p(hs92) t(year) asym
export delimited "`fixtures_dir'/stata_proximity_discrete_asym.csv", replace
di "  Saved: stata_proximity_discrete_asym.csv"

* Continuous proximity
import delimited "`input_file'", clear
genproximity export_val, i(origin) p(hs92) t(year) cont
export delimited "`fixtures_dir'/stata_proximity_continuous.csv", replace
di "  Saved: stata_proximity_continuous.csv"


/*******************************************************************************
 * STEP 11: Generate single-year fixture (edge case)
 ******************************************************************************/

di ""
di "Step 11: Generating single-year fixture..."

* Filter to just 1995
import delimited "`input_file'", clear
keep if year == 1995
ecomplexity export_val, i(origin) p(hs92) t(year)
export delimited "`fixtures_dir'/stata_single_year_1995.csv", replace
di "  Saved: stata_single_year_1995.csv"


/*******************************************************************************
 * Summary
 ******************************************************************************/

di ""
di "=============================================="
di "Fixture Generation Complete!"
di "=============================================="
di ""
di "ECOMPLEXITY FIXTURES (15 files):"
di "  1.  stata_ground_truth.csv         - Default (RCA=1, discrete, symmetric)"
di "  2.  stata_rca_05.csv               - RCA threshold = 0.5"
di "  3.  stata_rca_2.csv                - RCA threshold = 2.0"
di "  4.  stata_rpop_default.csv         - RPOP threshold = 1.0"
di "  5.  stata_rpop_2.csv               - RPOP threshold = 2.0"
di "  6.  stata_discrete_asymmetric.csv  - Asymmetric proximity"
di "  7.  stata_continuous.csv           - Continuous proximity (Pearson)"
di "  8.  stata_knn_5.csv                - KNN density (k=5)"
di "  9.  stata_knn_10.csv               - KNN density (k=10)"
di "  10. stata_knn_20.csv               - KNN density (k=20)"
di "  11. stata_manual_mcp.csv           - Manual MCP input"
di "  12. stata_rca_rpop_combined.csv    - Combined RCA + RPOP"
di "  13. stata_continuous_knn_10.csv    - Continuous + KNN=10"
di "  14. stata_continuous_rpop.csv      - Continuous + RPOP"
di "  15. stata_single_year_1995.csv     - Single year (edge case)"
di ""
di "PROXIMITY FIXTURES (3 files):"
di "  16. stata_proximity_discrete_sym.csv  - Discrete, symmetric"
di "  17. stata_proximity_discrete_asym.csv - Discrete, asymmetric"
di "  18. stata_proximity_continuous.csv    - Continuous (Pearson)"
di ""
di "SUPPORT FILES (1 file):"
di "  19. manual_mcp_sample.csv          - Pre-computed MCP for manual tests"
di ""
di "Total: 19 files generated"
di ""
di "Next step: Run Python validation tests:"
di "  uv run pytest tests/test_stata_validation.py -v"
di ""

exit, clear
