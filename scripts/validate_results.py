#!/usr/bin/env python3
"""Validate completed GBU-DEA reproducibility outputs and archive them."""
from pathlib import Path
import zipfile
import pandas as pd

main_dir = Path("results/GBU_DEA_outstanding_3x3_results")
random_dir = Path("results/random_partition_1000_results")
main_file = main_dir / "scenario_bank_level_scores.csv"
assert main_file.is_file(), f"Run the main analysis first: {main_file}"
main = pd.read_csv(main_file)
expected_main = 250 * 5 * 12 * 3
assert len(main) == expected_main, f"Expected {expected_main} main records; got {len(main)}"
assert set(main["model"]) == {"gbu", "bs", "local"}
assert set(main["delta"].unique()) == {0.01, 0.025, 0.05}
assert set(main["Gamma_G"].unique()) == {1, 2, 3, 5}
assert not main.duplicated(["frontier", "target_index", "delta", "Gamma_G", "model"]).any()
eq_file = main_dir / "proposition6_equality_bank_input.csv"
assert eq_file.is_file(), f"Missing Proposition 6 diagnostics: {eq_file}"
eq = pd.read_csv(eq_file)
assert len(eq) == expected_main
assert eq["equality_holds"].notna().all()
random_file = random_dir / "random_partition_level_results.csv"
assert random_file.is_file(), f"Run the random-partition experiment first: {random_file}"
random_results = pd.read_csv(random_file)
assert random_results["Partition"].nunique() == 1000, (
    f"Expected 1,000 completed partitions; got {random_results['Partition'].nunique()}"
)
archive = Path("results/GBU_DEA_reproducibility_outputs.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in (main_dir, random_dir):
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(Path("results"))))
print("Validation passed.")
print(f"Main-analysis records: {len(main):,}")
print(f"Proposition 6 records: {len(eq):,}")
print(f"Completed random partitions: {random_results['Partition'].nunique():,}")
print(f"Archive: {archive}")
