# Two-Level Grouped Budgeted Uncertainty for Robust DEA of European Banks

This repository contains a Python implementation of an input-oriented, variable-returns-to-scale (BCC) Data Envelopment Analysis model with a two-level grouped budgeted uncertainty set. The empirical application compares the proposed grouped-budgeted uncertainty (GBU) specification with a conventional single-level Bertsimas–Sim (BS) budgeted uncertainty specification.

The implementation reads a supplementary Excel workbook containing observations for 250 European banks in 23 countries. It estimates conventional group and pooled metafrontier scores, robust scores under a grid of uncertainty scenarios, and a set of score, ranking, monotonicity, and support-function diagnostics.

## Repository contents

```text
.
├── README.md
├── GDEA_GitHub.ipynb
├── run_gbu_dea.py
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── results/
    └── .gitkeep
```

- `run_gbu_dea.py`: end-to-end implementation of the empirical analysis.
- `GDEA_GitHub.ipynb`: notebook interface to the analysis.
- `requirements.txt`: Python package requirements.
- `data/README.md`: instructions for obtaining and placing the source workbook.
- `results/`: destination for generated tables and score files. Generated results are not committed by default.

## Data

The analysis expects the following Excel workbook:

```text
1-s2.0-S2352340918314410-mmc2(1).xlsx
```

Place it in the repository's `data/` directory:

```text
data/1-s2.0-S2352340918314410-mmc2(1).xlsx
```

The script reads the first worksheet. It expects 250 observations and the bank and country identifiers, together with the following variables (column names are normalized for surrounding whitespace):

**Inputs**

1. `Employees (I)`
2. `Assets (I)`
3. `Personnel Expenses (I)`

**Outputs**

1. `Loans (O)`
2. `Net Income Revenue (O)`
3. `Net Fees Commission (O)`

The source data are not included in this repository. Obtain the workbook from its original source and check its citation and redistribution terms before using or sharing it. Do not upload bank-level data to a public repository unless redistribution is permitted.

## Empirical grouping

The 250 banks are assigned to four regional DEA frontiers. The regional sample sizes used by the script are:

| Regional frontier | Number of banks |
|---|---:|
| Eastern Europe | 22 |
| Northern Europe | 33 |
| Southern Europe | 66 |
| Western Europe | 129 |
| **Total** | **250** |

The pooled metafrontier uses all 250 banks. The uncertainty groups are **countries**, not regions. Thus, each regional frontier contains the country groups belonging to that region, while the pooled metafrontier contains all 23 country groups.

The country-to-region mapping is explicitly listed in `run_gbu_dea.py`. It is a reconstructed mapping and should be checked against the original study's supplementary documentation before the regional classification is used in a publication.

## Model outline

For each evaluated bank, the model minimizes the input contraction factor \(\theta\), subject to output requirements and variable-returns-to-scale convexity. In the robust input constraints, only the **reference inputs** are uncertain. The evaluated bank's input and output data, and all output coefficients, are held fixed.

For input \(k\), the robust constraint has the form

\[
\sum_j x_{kj}\lambda_j + \delta\,\beta_{\mathrm{GBU}}\big((x_{kj}\lambda_j)_j\big)
\leq \theta x_{k0},
\]

with the output constraints

\[
\sum_j y_{rj}\lambda_j \geq y_{r0},\qquad
\sum_j\lambda_j=1,\qquad \lambda_j\geq 0.
\]

The GBU support function uses a local budget within each country and a global budget across country-level activations. Its linear programming representation is incorporated into the BCC model.

The script also solves a single-level BS comparator. The BS model uses the same relative uncertainty radius and the same numerical value of the global budget, but applies one flat budget over reference banks. The GBU and BS uncertainty sets therefore have different structures; the comparison is between these two uncertainty specifications, not between identical uncertainty sets.

## Scenario settings

The default analysis evaluates the following 12 combinations:

| Parameter | Values |
|---|---|
| Relative upward reference-input uncertainty, \(\delta\) | 0.01, 0.025, 0.05 |
| Local country budget, \(\Gamma_{\mathrm{local}}\) | 1 |
| Global activation budget, \(\Gamma_{\mathrm{global}}\) | 1, 2, 3, 5 |

The conventional BCC scores are also computed as the zero-uncertainty baseline. The default setting `QUICK_TEST = False` evaluates all 250 banks. A quick test can be enabled in the script for debugging, but its partial-sample results must not be reported as full-sample empirical results.

## Installation

Python 3.10 or later is recommended. From the repository root, create and activate a virtual environment if desired, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run the analysis

After placing the source workbook in `data/`, run:

```bash
python run_gbu_dea.py
```

The optimization problems are solved using `scipy.optimize.linprog` with the HiGHS method. The script creates the output directory automatically.

To run the notebook instead, open `GDEA_GitHub.ipynb` in Jupyter or JupyterLab and execute its cells in order. The script is the recommended entry point for a command-line run.

## Generated outputs

The script writes its results to:

```text
results/TwoLevel_GBU_DEA_Results/
```

Principal files include:

- `TwoLevel_GBU_Empirical_Results.xlsx`: workbook containing bank-level scores and summary/diagnostic tables.
- `TwoLevel_GBU_Bank_Scores.csv`: bank-level conventional and robust scores.
- `GBU_vs_BS_Summary.csv`: aggregate comparison of GBU and single-level BS scores.
- `GBU_vs_BS_Bank_Level.csv`: bank-level paired GBU–BS score differences.
- `Overall_Summary.csv` and `Regional_Summary.csv`: overall and regional summaries.
- `Ranking_Stability.csv`: ranking comparisons across conventional and robust scores.
- `Monotonicity.csv`: diagnostics for score changes as the global budget increases.
- `Country_Heterogeneity.csv`: country-level summaries of paired GBU–BS differences.
- `GBU_BS_Ranking_Overlap.csv`: ranking correlation and best/worst decile overlap for GBU and BS.
- `Country_Cluster_Bootstrap.csv`: country-cluster bootstrap intervals for paired score differences.
- `Proposition6_Cases.csv` and `Proposition6_Equality_Frequency.csv`: numerical support-function checks associated with Proposition 6.

The Excel workbook also contains the corresponding tables as separate worksheets.

## Bootstrap interpretation

The country-cluster bootstrap resamples countries and calculates intervals for the **observed paired bank-level score differences**. The DEA frontiers are not re-estimated within each bootstrap replication. These intervals therefore describe resampling variability of the fixed estimated score differences; they are not a full bootstrap that repeats the DEA estimation procedure.

The random seed and number of replications are set in `run_gbu_dea.py` (`BOOTSTRAP_SEED` and `BOOTSTRAP_REPS`). The default is 5,000 replications.

## Reproducibility notes and limitations

1. Results depend on the exact source workbook and its first worksheet. Check that the workbook has the expected columns and observations.
2. The regional country mapping is reconstructed in the code and requires verification against the source documentation.
3. The primary specification uses three inputs and three outputs as listed above. The script does not silently substitute the alternative variable specification from earlier exploratory notebook cells.
4. The robust perturbations are upward-only and apply to reference inputs. Outputs and target-bank observations are fixed.
5. The GBU and BS models use different uncertainty geometries. A numerical comparison at the same values of \(\delta\) and \(\Gamma_{\mathrm{global}}\) does not mean that their uncertainty sets coincide.
6. The country-cluster bootstrap does not re-estimate the DEA frontiers in each replication.
7. The code produces numerical results from the supplied workbook; this repository does not contain precomputed empirical results or the source data.

## Citation

If you use this code, please cite the associated research manuscript. A formal citation can be added here once the manuscript's publication details are available. Please also cite the original source of the bank-level supplementary dataset according to its documentation.

## License

No software license is currently specified. Unless a license is added to this repository, reuse, modification, and redistribution are not granted by an explicit open-source license. Add a license only after the repository owner has selected the intended terms.
