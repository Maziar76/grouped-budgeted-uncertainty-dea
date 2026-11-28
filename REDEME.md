# Grouped Budgeted Uncertainty DEA — reproducibility package

This repository provides code for the revised 3-input/3-output European-bank application in the manuscript. It includes the main 12-scenario comparison and the separate 1,000 within-region random-partition experiment.

## Repository contents

```text
.
├── GDEA_reproducibility.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── results/
│   └── .gitkeep
└── scripts/
    ├── run_main_analysis.py
    ├── run_random_partitions.py
    └── validate_results.py
```

## Data requirement

The source workbook is **not included**. Obtain `1-s2.0-S2352340918314410-mmc2.xlsx` from the supplementary materials of the source publication, check the applicable terms, and place it at:

```text
data/1-s2.0-S2352340918314410-mmc2.xlsx
```

The code expects the workbook columns named in the scripts, including `BANKS`, `COUNTRY`, the three selected input columns, and the three selected output columns. It checks the expected 250 observations and regional sample sizes before estimation.

## Environment

Python 3.10 or later is recommended. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The LPs are solved with SciPy's `linprog` interface to HiGHS; no proprietary optimization solver is required.

## Reproduction steps

From the repository root, run:

```bash
python scripts/run_main_analysis.py
python scripts/run_random_partitions.py
python scripts/validate_results.py
```

Alternatively, open `GDEA_reproducibility.ipynb` and execute its cells in order. The main analysis evaluates all combinations of `delta = {0.01, 0.025, 0.05}` and `Gamma_G = {1, 2, 3, 5}` for GBU, product-calibrated flat BS, and local-only GBU. The random-partition analysis is a separate computationally intensive experiment configured for 1,000 replications and supports checkpoint-based resumption.

## Outputs

Generated outputs are saved under `results/`. The validation script checks the expected main-analysis and Proposition 6 record counts, verifies that 1,000 distinct random partitions are present, and creates `results/GBU_DEA_reproducibility_outputs.zip`. Generated outputs are not committed by default.

## Important reproducibility qualifications

- The country-to-region assignment is encoded explicitly in the scripts. It must be checked against the original supplementary documentation before being described as independently verified.
- The Proposition 6 equality diagnostic is evaluated using solver-selected optimal intensity vectors. If an LP has multiple optimal intensity vectors, the reported diagnostic can depend on the selected optimum.
- The random-partition experiment preserves the within-region multiset of country-group sizes. Its reported empirical quantile ranges are not bootstrap confidence intervals.
- The workbook is not redistributed by this repository. Users are responsible for obtaining it lawfully and complying with its terms.

## Suggested citation

If you use this code, cite the associated manuscript: *Grouped Budgeted Uncertainty DEA: A Two-Level Robust BCC Efficiency Model*. Add the final bibliographic details after publication.
