#!/usr/bin/env python3
"""Run the 1,000 matched-size random-partition experiment for GBU-DEA.

Place the supplementary workbook in data/ or set INPUT_FILE below.

The script:
  * uses 3 inputs and 3 outputs (see COLUMN_MAP below);
  * assigns the 23 countries to the four manuscript regions;
  * preserves the number and exact multiset of country-group sizes in each region;
  * recomputes regional and pooled input-oriented VRS/BCC and GBU scores;
  * calculates Spearman and Kendall tau-b correlations with BCC scores;
  * checkpoints results and can resume after interruption.

Install: pip install numpy pandas scipy openpyxl
Run:     python GBU_DEA_random_partitions_1000_STABLE.py

Do not change the variable specification or regional mapping without checking
it against the manuscript and the data source.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import kendalltau, spearmanr

# ============================= USER SETTINGS =============================

# Expected source workbook (not distributed with this repository).
INPUT_FILE = "data/1-s2.0-S2352340918314410-mmc2.xlsx"
SHEET_NAME = 0

# Exact headings in the uploaded supplementary workbook.
COLUMN_MAP = {
    "bank": "BANKS",
    "country": "COUNTRY",
    "employees": " Employees (I)",
    "assets": "Assets (I)",
    "personnel": "Personnel Expenses (I)",
    "loans": "Loans (O)",
    "net_income_revenue": "Net Income Revenue (O)",
    "fees_commissions": "Net Fees Commission (O)",
}

N_RANDOM_PARTITIONS = 1000
RANDOM_SEED = 20261127
DELTA = 0.05
GAMMA_LOCAL = 1.0
GAMMA_GLOBAL = 5.0

# Set to 1 for maximum notebook compatibility. If running this as a .py file
# on Linux and the environment supports multiprocessing, 2 or 4 can be tried.
N_WORKERS = 1
CHECKPOINT_EVERY = 5
OUTPUT_DIR = "results/random_partition_1000_results"

INPUT_COLS = ["employees", "assets", "personnel"]
OUTPUT_COLS = ["loans", "net_income_revenue", "fees_commissions"]

COUNTRY_TO_REGION = {
    # Eastern Europe
    "Czech Republic": "Eastern Europe",
    "Hungary": "Eastern Europe",
    "Poland": "Eastern Europe",
    "Romania": "Eastern Europe",
    "Slovakia": "Eastern Europe",
    "Slovenia": "Eastern Europe",
    # Northern Europe
    "Denmark": "Northern Europe",
    "Finland": "Northern Europe",
    "Ireland": "Northern Europe",
    "Sweden": "Northern Europe",
    "United Kingdom": "Northern Europe",
    # Southern Europe
    "Croatia": "Southern Europe",
    "Cyprus": "Southern Europe",
    "Greece": "Southern Europe",
    "Italy": "Southern Europe",
    "Portugal": "Southern Europe",
    "Spain": "Southern Europe",
    # Western Europe
    "Austria": "Western Europe",
    "Belgium": "Western Europe",
    "France": "Western Europe",
    "Germany": "Western Europe",
    "Luxembourg": "Western Europe",
    "Netherlands": "Western Europe",
}
EXPECTED_REGION_SIZES = {
    "Eastern Europe": 22,
    "Northern Europe": 33,
    "Southern Europe": 66,
    "Western Europe": 129,
}

# ============================= FILE DISCOVERY =============================

def locate_input_file() -> Path:
    """Locate the source workbook in the project data/ folder or working directory."""
    requested = Path(INPUT_FILE)
    candidates = [requested, Path.cwd() / requested.name,
                  Path("data") / requested.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(Path("data").glob("*S2352340918314410*mmc2*.xlsx"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            "Multiple source workbooks found in data/. Set INPUT_FILE explicitly."
        )
    raise FileNotFoundError(
        f"Source workbook not found: {requested}. Place it in data/ or update INPUT_FILE."
    )


# ============================= DATA LOADING =============================

def read_raw_data() -> tuple[pd.DataFrame, Path]:
    path = locate_input_file()
    raw = pd.read_excel(path, sheet_name=SHEET_NAME)

    missing = [source for source in COLUMN_MAP.values() if source not in raw.columns]
    if missing:
        raise ValueError(
            "Required source columns were not found:\n  - "
            + "\n  - ".join(missing)
            + "\nActual workbook columns:\n  - "
            + "\n  - ".join(map(str, raw.columns))
        )

    df = pd.DataFrame({key: raw[source] for key, source in COLUMN_MAP.items()})
    df = df.dropna(subset=list(COLUMN_MAP.keys())).copy()
    df["country"] = df["country"].astype(str).str.strip()
    df["bank"] = df["bank"].astype(str).str.strip()

    unknown = sorted(set(df["country"]) - set(COUNTRY_TO_REGION))
    if unknown:
        raise ValueError("Country-region assignment missing for: " + ", ".join(unknown))
    df["region"] = df["country"].map(COUNTRY_TO_REGION)

    for col in INPUT_COLS + OUTPUT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[INPUT_COLS + OUTPUT_COLS].isna().any().any():
        bad = df.loc[df[INPUT_COLS + OUTPUT_COLS].isna().any(axis=1)]
        raise ValueError("Non-numeric or missing DEA data found:\n" + bad.head(10).to_string())
    if (df[INPUT_COLS + OUTPUT_COLS] <= 0).any().any():
        badcols = [c for c in INPUT_COLS + OUTPUT_COLS if (df[c] <= 0).any()]
        raise ValueError(f"All DEA data must be strictly positive; check {badcols}.")

    observed = df["region"].value_counts().to_dict()
    if observed != EXPECTED_REGION_SIZES:
        raise ValueError(
            "Region sample sizes do not match the manuscript. "
            f"Observed={observed}; expected={EXPECTED_REGION_SIZES}. "
            "Do not proceed until the country allocation is checked."
        )

    df = df.reset_index(drop=True)
    df["bank_id"] = np.arange(len(df), dtype=int)
    return df, path

# ============================= DEA MODELS =============================

def solve_bcc_score(X: np.ndarray, Y: np.ndarray, target: int) -> float:
    """Input-oriented VRS/BCC envelopment score."""
    n, m = X.shape
    s = Y.shape[1]
    c = np.zeros(1 + n)
    c[0] = 1.0
    A_ub, b_ub = [], []

    for i in range(m):
        row = np.zeros(1 + n)
        row[0] = -X[target, i]
        row[1:] = X[:, i]
        A_ub.append(row)
        b_ub.append(0.0)
    for r in range(s):
        row = np.zeros(1 + n)
        row[1:] = -Y[:, r]
        A_ub.append(row)
        b_ub.append(-Y[target, r])

    A_eq = np.zeros((1, 1 + n))
    A_eq[0, 1:] = 1.0
    res = linprog(
        c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
        A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (1 + n), method="highs"
    )
    if not res.success:
        raise RuntimeError(f"BCC LP failed for target {target}: {res.message}")
    return float(res.fun)


def solve_gbu_score(
    X: np.ndarray,
    Y: np.ndarray,
    target: int,
    group_codes: np.ndarray,
    delta: float = DELTA,
    gamma_local: float = GAMMA_LOCAL,
    gamma_global: float = GAMMA_GLOBAL,
) -> float:
    """Input-oriented VRS GBU score using the nested-budget support LP dual."""
    # Positive column scaling is exactly score-invariant in input-oriented
    # DEA: multiplying any input/output coordinate by a positive constant
    # multiplies both sides of its corresponding constraint by that constant.
    # It is nevertheless important numerically here because the raw workbook
    # mixes variables ranging from single digits to nearly two million.  A
    # few random partitions trigger a HiGHS status-15 numerical failure unless
    # each coordinate is normalized before building the nested-budget LP.
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_scale = np.max(np.abs(X), axis=0)
    y_scale = np.max(np.abs(Y), axis=0)
    if np.any(x_scale <= 0) or np.any(y_scale <= 0):
        raise ValueError("GBU DEA inputs and outputs must have positive column scales")
    X = X / x_scale
    Y = Y / y_scale

    n, m = X.shape
    s = Y.shape[1]
    groups = np.asarray(group_codes)
    if groups.shape != (n,):
        raise ValueError("group_codes must have one entry per reference DMU")
    _, groups = np.unique(groups, return_inverse=True)
    K = int(groups.max()) + 1

    # Variables: theta, lambda[n], then for each input i: rho, v[K], pi[K], w[n].
    pos = 0
    theta_idx = pos
    pos += 1
    lam_idx = np.arange(pos, pos + n)
    pos += n
    rho_idx, v_idx, pi_idx, w_idx = [], [], [], []
    for _ in range(m):
        rho_idx.append(pos); pos += 1
        v_idx.append(np.arange(pos, pos + K)); pos += K
        pi_idx.append(np.arange(pos, pos + K)); pos += K
        w_idx.append(np.arange(pos, pos + n)); pos += n

    nvar = pos
    c = np.zeros(nvar)
    c[theta_idx] = 1.0
    A_ub, b_ub = [], []

    for i in range(m):
        # Nominal input use plus the nested-budget support function <= theta*x_io.
        row = np.zeros(nvar)
        row[theta_idx] = -X[target, i]
        row[lam_idx] = X[:, i]
        row[rho_idx[i]] = gamma_global
        row[v_idx[i]] = 1.0
        A_ub.append(row); b_ub.append(0.0)

        # v_g >= Gamma_local*pi_g + sum_{j in g} w_j - rho.
        for g in range(K):
            row = np.zeros(nvar)
            row[pi_idx[i][g]] = gamma_local
            row[w_idx[i][groups == g]] = 1.0
            row[rho_idx[i]] = -1.0
            row[v_idx[i][g]] = -1.0
            A_ub.append(row); b_ub.append(0.0)

        # w_j >= delta*X_ji*lambda_j - pi_g.
        for j in range(n):
            row = np.zeros(nvar)
            row[lam_idx[j]] = delta * X[j, i]
            row[pi_idx[i][groups[j]]] = -1.0
            row[w_idx[i][j]] = -1.0
            A_ub.append(row); b_ub.append(0.0)

    # Nominal output constraints.
    for r in range(s):
        row = np.zeros(nvar)
        row[lam_idx] = -Y[:, r]
        A_ub.append(row); b_ub.append(-Y[target, r])

    A_eq = np.zeros((1, nvar))
    A_eq[0, lam_idx] = 1.0
    res = linprog(
        c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
        A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * nvar, method="highs"
    )
    if not res.success:
        # A dual-simplex retry is useful for residual HiGHS presolve/status
        # issues.  The model is unchanged; only the algorithm is different.
        retry = linprog(
            c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
            A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * nvar,
            method="highs-ds",
            options={"presolve": False, "dual_feasibility_tolerance": 1e-8,
                     "primal_feasibility_tolerance": 1e-8},
        )
        if retry.success:
            res = retry
        else:
            raise RuntimeError(
                f"GBU LP failed for target {target}; primary={res.message}; "
                f"dual-simplex retry={retry.message}"
            )
    if not np.isfinite(res.fun):
        raise RuntimeError(f"GBU LP returned a non-finite objective for target {target}")
    return float(res.fun)

# ============================= PARTITIONS =============================

def build_country_groups(df: pd.DataFrame) -> dict:
    info = {}
    for region, sub in df.groupby("region", sort=True):
        counts = sub.groupby("country", sort=True).size().sort_values(ascending=False)
        info[region] = {
            "bank_ids": sub["bank_id"].to_numpy(dtype=int),
            "countries": counts.index.tolist(),
            "sizes": counts.to_numpy(dtype=int),
        }
    return info


def make_country_codes(df: pd.DataFrame) -> np.ndarray:
    pairs = list(zip(df["region"].astype(str), df["country"].astype(str)))
    lookup = {pair: i for i, pair in enumerate(sorted(set(pairs)))}
    return np.array([lookup[pair] for pair in pairs], dtype=int)


def random_partition_codes(df: pd.DataFrame, region_info: dict, rng: np.random.Generator) -> np.ndarray:
    """Shuffle banks within each region into groups of the country group sizes."""
    codes = np.full(len(df), -1, dtype=int)
    next_code = 0
    for region in sorted(region_info):
        ids = region_info[region]["bank_ids"]
        sizes = region_info[region]["sizes"]
        shuffled = rng.permutation(ids)
        start = 0
        for size in sizes:
            members = shuffled[start:start + int(size)]
            codes[members] = next_code
            next_code += 1
            start += int(size)
        if start != len(shuffled):
            raise RuntimeError(f"Partition construction failed in {region}")
    if np.any(codes < 0):
        raise RuntimeError("At least one bank was not assigned to a random group")
    return codes

# ============================= SCORING =============================

def stable_rank_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman and Kendall tau-b after suppressing solver-level noise.

    DEA LPs can return mathematically tied scores with tiny floating-point
    differences (for example 1.0 versus 0.9999999999999998). The earlier
    32-partition analysis used scores rounded to 10 decimal places for ranks;
    applying the same convention here keeps the 1,000-partition results
    comparable and prevents numerical noise from breaking genuine ties.
    """
    xr = np.round(np.asarray(x, dtype=float), 10)
    yr = np.round(np.asarray(y, dtype=float), 10)
    return (
        float(spearmanr(xr, yr).statistic),
        float(kendalltau(xr, yr, variant="b").statistic),
    )


def score_one_partition(partition_id, df, X, Y, random_codes, bcc_regional, bcc_meta):
    n = len(df)
    region_values = df["region"].to_numpy()
    regional_scores = np.full(n, np.nan)
    meta_scores = np.full(n, np.nan)

    # Regional frontiers.
    for region in sorted(df["region"].unique()):
        ids = np.flatnonzero(region_values == region)
        Xr, Yr, gr = X[ids], Y[ids], random_codes[ids]
        for local_target, global_target in enumerate(ids):
            regional_scores[global_target] = solve_gbu_score(Xr, Yr, local_target, gr)

    # Pooled metafrontier; group codes are globally unique across regions.
    for target in range(n):
        meta_scores[target] = solve_gbu_score(X, Y, target, random_codes)

    region_rows = []
    for region in sorted(df["region"].unique()):
        ids = np.flatnonzero(region_values == region)
        sp, kt = stable_rank_correlations(regional_scores[ids], bcc_regional[ids])
        region_rows.append({
            "Partition": partition_id,
            "Region": region,
            "N_Banks": len(ids),
            "Regional_GBU_Mean": float(np.mean(regional_scores[ids])),
            "Regional_Spearman_vs_BCC": sp,
            "Regional_Kendall_tau_b_vs_BCC": kt,
        })
    reg = pd.DataFrame(region_rows)

    meta_sp, meta_kt = stable_rank_correlations(meta_scores, bcc_meta)
    summary = {
        "Partition": partition_id,
        "Regional_GBU_Mean_All_Banks": float(np.mean(regional_scores)),
        "Pooled_Meta_GBU_Mean": float(np.mean(meta_scores)),
        "Mean_Regional_Spearman_vs_BCC": float(reg["Regional_Spearman_vs_BCC"].mean()),
        "Mean_Regional_Kendall_tau_b_vs_BCC": float(reg["Regional_Kendall_tau_b_vs_BCC"].mean()),
        "Meta_Spearman_vs_BCC": meta_sp,
        "Meta_Kendall_tau_b_vs_BCC": meta_kt,
    }
    bank = pd.DataFrame({
        "Partition": partition_id,
        "bank_id": df["bank_id"].to_numpy(),
        "BANKS": df["bank"].astype(str).to_numpy(),
        "COUNTRY": df["country"].astype(str).to_numpy(),
        "REGION": region_values,
        "Random_Group_Code": random_codes,
        "Regional_GBU": regional_scores,
        "Meta_GBU": meta_scores,
    })
    return summary, region_rows, bank

# ============================= OUTPUT / RESUME =============================

def append_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_completed(outdir: Path) -> set[int]:
    p = outdir / "random_partition_level_results.csv"
    if not p.exists():
        return set()
    try:
        return set(pd.read_csv(p, usecols=["Partition"])["Partition"].astype(int).tolist())
    except Exception as exc:
        raise RuntimeError(f"Cannot read existing checkpoint {p}: {exc}")


def write_country_baseline(outdir, df, X, Y, country_codes, bcc_regional, bcc_meta):
    summary_path = outdir / "country_partition_summary.csv"
    bank_path = outdir / "country_partition_bank_scores.csv"
    region_path = outdir / "country_partition_region_results.csv"
    if summary_path.exists() and bank_path.exists() and region_path.exists():
        return pd.read_csv(summary_path).iloc[0].to_dict()
    summary, region_rows, bank = score_one_partition(
        0, df, X, Y, country_codes, bcc_regional, bcc_meta
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(region_rows).to_csv(region_path, index=False)
    bank = bank.rename(columns={
        "Random_Group_Code": "Country_Group_Code",
        "Regional_GBU": "Country_Regional_GBU",
        "Meta_GBU": "Country_Meta_GBU",
    })
    bank["BCC_Regional"] = bcc_regional
    bank["BCC_Meta"] = bcc_meta
    bank.to_csv(bank_path, index=False)
    return summary


def finalize_outputs(outdir: Path, country_summary: dict, successful_count: int):
    part_path = outdir / "random_partition_level_results.csv"
    reg_path = outdir / "random_partition_region_results.csv"
    bank_path = outdir / "random_partition_bank_level_scores.csv"
    if not part_path.exists():
        print("No random partition results have been saved yet.")
        return
    part = pd.read_csv(part_path).sort_values("Partition")
    reg = pd.read_csv(reg_path).sort_values(["Partition", "Region"]) if reg_path.exists() else pd.DataFrame()

    # Rebuild comparison table from all successful partitions.
    metrics = [
        "Regional_GBU_Mean_All_Banks",
        "Pooled_Meta_GBU_Mean",
        "Mean_Regional_Spearman_vs_BCC",
        "Mean_Regional_Kendall_tau_b_vs_BCC",
        "Meta_Spearman_vs_BCC",
        "Meta_Kendall_tau_b_vs_BCC",
    ]
    rows = []
    B = len(part)
    for metric in metrics:
        observed = float(country_summary[metric])
        vals = part[metric].dropna().to_numpy(dtype=float)
        b = int(np.sum(vals <= observed + 1e-12))
        rows.append({
            "Metric": metric,
            "Country_Partition": observed,
            "Random_Mean": float(np.mean(vals)),
            "Random_SD": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "Random_P05": float(np.quantile(vals, 0.05)),
            "Random_Median": float(np.quantile(vals, 0.50)),
            "Random_P95": float(np.quantile(vals, 0.95)),
            "Random_Min": float(np.min(vals)),
            "Random_Max": float(np.max(vals)),
            "B_Random_Partitions": B,
            "Count_Random_LE_Country": b,
            "Monte_Carlo_Lower_Tail_p": (1 + b) / (B + 1),
        })
    pd.DataFrame(rows).to_csv(outdir / "random_partition_comparison_summary.csv", index=False)

    # Bank-wise distribution, built from the complete bank-level score file.
    if bank_path.exists():
        bank = pd.read_csv(bank_path)
        dist = bank.groupby(["bank_id", "BANKS", "COUNTRY", "REGION"], as_index=False).agg(
            Random_Regional_Mean=("Regional_GBU", "mean"),
            Random_Regional_SD=("Regional_GBU", "std"),
            Random_Regional_P05=("Regional_GBU", lambda x: x.quantile(.05)),
            Random_Regional_Median=("Regional_GBU", "median"),
            Random_Regional_P95=("Regional_GBU", lambda x: x.quantile(.95)),
            Random_Meta_Mean=("Meta_GBU", "mean"),
            Random_Meta_SD=("Meta_GBU", "std"),
            Random_Meta_P05=("Meta_GBU", lambda x: x.quantile(.05)),
            Random_Meta_Median=("Meta_GBU", "median"),
            Random_Meta_P95=("Meta_GBU", lambda x: x.quantile(.95)),
        )
        country = pd.read_csv(outdir / "country_partition_bank_scores.csv")
        dist = dist.merge(country[["bank_id", "Country_Regional_GBU", "Country_Meta_GBU", "BCC_Regional", "BCC_Meta"]], on="bank_id", how="left")
        dist.to_csv(outdir / "random_partition_bank_distribution.csv", index=False)

    print("\nCurrent checkpoint summary:")
    print(f"  successful random partitions: {successful_count}")
    print(f"  regional result rows: {len(reg)}")
    print(f"  comparison table: {outdir / 'random_partition_comparison_summary.csv'}")

# ============================= MAIN =============================

def main():
    t0 = time.time()
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df, source_path = read_raw_data()
    X = df[INPUT_COLS].to_numpy(dtype=float)
    Y = df[OUTPUT_COLS].to_numpy(dtype=float)
    n = len(df)
    region_info = build_country_groups(df)
    country_codes = make_country_codes(df)

    print("=" * 76)
    print("GBU-DEA RANDOM-PARTITION EXPERIMENT")
    print(f"Source workbook: {source_path}")
    print(f"DMUs: {n}; regions: {len(region_info)}; country groups: {len(np.unique(country_codes))}")
    for region, info in region_info.items():
        print(f"  {region}: {len(info['bank_ids'])} banks; group sizes={list(map(int, info['sizes']))}")
    print(f"Inputs: {COLUMN_MAP['employees']}; {COLUMN_MAP['assets']}; {COLUMN_MAP['personnel']}")
    print(f"Outputs: {COLUMN_MAP['loans']}; {COLUMN_MAP['net_income_revenue']}; {COLUMN_MAP['fees_commissions']}")
    print(f"Random partitions={N_RANDOM_PARTITIONS}; seed={RANDOM_SEED}; delta={DELTA}; Gamma_local={GAMMA_LOCAL}; Gamma_global={GAMMA_GLOBAL}")
    print(f"Output folder: {outdir.resolve()}")
    print("=" * 76)

    # Conventional BCC reference scores are computed once and checkpointed.
    bcc_file = outdir / "bcc_reference_scores.csv"
    if bcc_file.exists():
        bcc = pd.read_csv(bcc_file).sort_values("bank_id")
        if len(bcc) != n:
            raise RuntimeError("Existing BCC checkpoint has a different number of banks. Remove the output folder to restart cleanly.")
        bcc_regional = bcc["BCC_Regional"].to_numpy(float)
        bcc_meta = bcc["BCC_Meta"].to_numpy(float)
        print("Loaded cached BCC reference scores.")
    else:
        print("Computing conventional regional BCC scores...")
        bcc_regional = np.full(n, np.nan)
        for region in sorted(df["region"].unique()):
            ids = np.flatnonzero(df["region"].to_numpy() == region)
            for local, global_id in enumerate(ids):
                bcc_regional[global_id] = solve_bcc_score(X[ids], Y[ids], local)
        print("Computing conventional pooled BCC scores...")
        bcc_meta = np.array([solve_bcc_score(X, Y, j) for j in range(n)])
        pd.DataFrame({"bank_id": df["bank_id"], "BCC_Regional": bcc_regional, "BCC_Meta": bcc_meta}).to_csv(bcc_file, index=False)

    country_summary = write_country_baseline(outdir, df, X, Y, country_codes, bcc_regional, bcc_meta)
    completed_ids = load_completed(outdir)
    print(f"Country baseline ready. Existing random partitions to resume: {len(completed_ids)}")

    # Generate every partition deterministically from the seed; skipped IDs are
    # still generated so that resumed runs reproduce exactly the same sequence.
    rng = np.random.default_rng(RANDOM_SEED)
    pending = []
    for pid in range(1, N_RANDOM_PARTITIONS + 1):
        codes = random_partition_codes(df, region_info, rng)
        if pid not in completed_ids:
            pending.append((pid, codes))

    payload = {
        "df": df, "X": X, "Y": Y,
        "bcc_regional": bcc_regional, "bcc_meta": bcc_meta,
    }
    print(f"Partitions remaining: {len(pending)}")
    if N_WORKERS == 1:
        global _WORKER
        _WORKER = payload
        results_iter = (_run_one_task(task) for task in pending)
        batch_summaries, batch_regions, batch_banks = [], [], []
        done = 0
        for result in results_iter:
            summary, region_rows, bank = result
            batch_summaries.append(summary)
            batch_regions.extend(region_rows)
            batch_banks.append(bank)
            done += 1
            if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                append_frame(pd.DataFrame(batch_summaries), outdir / "random_partition_level_results.csv")
                append_frame(pd.DataFrame(batch_regions), outdir / "random_partition_region_results.csv")
                append_frame(pd.concat(batch_banks, ignore_index=True), outdir / "random_partition_bank_level_scores.csv")
                completed_ids.update(int(x["Partition"]) for x in batch_summaries)
                print(f"Saved checkpoint: {len(completed_ids)}/{N_RANDOM_PARTITIONS} partitions complete; elapsed {(time.time()-t0)/60:.1f} min")
                batch_summaries, batch_regions, batch_banks = [], [], []
    else:
        # Multiprocessing is supported when the script is executed as a .py file.
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(payload,)) as pool:
            futures = {pool.submit(_run_one_task, task): task[0] for task in pending}
            batch_summaries, batch_regions, batch_banks = [], [], []
            done = 0
            for future in as_completed(futures):
                summary, region_rows, bank = future.result()
                batch_summaries.append(summary)
                batch_regions.extend(region_rows)
                batch_banks.append(bank)
                done += 1
                if done % CHECKPOINT_EVERY == 0 or done == len(pending):
                    append_frame(pd.DataFrame(batch_summaries), outdir / "random_partition_level_results.csv")
                    append_frame(pd.DataFrame(batch_regions), outdir / "random_partition_region_results.csv")
                    append_frame(pd.concat(batch_banks, ignore_index=True), outdir / "random_partition_bank_level_scores.csv")
                    completed_ids.update(int(x["Partition"]) for x in batch_summaries)
                    print(f"Saved checkpoint: {len(completed_ids)}/{N_RANDOM_PARTITIONS} partitions complete; elapsed {(time.time()-t0)/60:.1f} min")
                    batch_summaries, batch_regions, batch_banks = [], [], []

    # Failed or interrupted runs can be restarted: already completed partition
    # IDs are skipped and the same deterministic partition sequence is used.
    completed_ids = load_completed(outdir)
    finalize_outputs(outdir, country_summary, len(completed_ids))

    config = [
        f"source_workbook={source_path}",
        f"n_banks={n}",
        f"regions={len(region_info)}",
        f"country_groups={len(np.unique(country_codes))}",
        f"requested_random_partitions={N_RANDOM_PARTITIONS}",
        f"successful_random_partitions={len(completed_ids)}",
        f"random_seed={RANDOM_SEED}",
        f"delta={DELTA}",
        f"gamma_local={GAMMA_LOCAL}",
        f"gamma_global={GAMMA_GLOBAL}",
        f"workers={N_WORKERS}",
        "randomization=within each region, preserving the multiset of country-group sizes",
        "regional_frontier=reference banks from the target region",
        "metafrontier=all banks; group identities remain distinct across regions",
        "orientation=input; returns_to_scale=VRS/BCC",
        "inputs=Employees, Assets, Personnel Expenses",
        "outputs=Loans, Net Income Revenue, Net Fees Commission",
        "rank_correlations=Spearman and Kendall tau-b computed after rounding scores to 10 decimal places to suppress solver-level numerical noise",
    ]
    (outdir / "run_configuration.txt").write_text("\n".join(config) + "\n", encoding="utf-8")
    print(f"\nRun finished or checkpointed in {(time.time()-t0)/60:.1f} minutes.")
    print(f"Completed partitions: {len(completed_ids)}/{N_RANDOM_PARTITIONS}")
    print(f"Results are in: {outdir.resolve()}")
    if len(completed_ids) < N_RANDOM_PARTITIONS:
        print("To continue after interruption, run this same script again; it will resume.")

# Worker definitions kept at module level for multiprocessing compatibility.
_WORKER = {}

def _init_worker(payload):
    global _WORKER
    _WORKER = payload

def _run_one_task(task):
    partition_id, codes = task
    p = _WORKER
    return score_one_partition(
        partition_id, p["df"], p["X"], p["Y"], codes,
        p["bcc_regional"], p["bcc_meta"]
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed checkpoints are safe; rerun to resume.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
