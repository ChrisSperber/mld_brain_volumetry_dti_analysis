"""Analyse brain imaging markers with mixed linear models.

Use mixed linear models to analyse the relationship between brain imaging markers and Gross Motor
Function Scores (GMFC). Each marker set (for varying metrics FA/MD/volumetry and varying resolution
level region/structure/whole_brain) is analysed with a mass-univariate test seperately for each
variable. Computed p-values are not corrected for multiple comparisons in the output data.

In each test, the role of the brain image marker is tested by comparison of two mixed models via
likelihood ratio test (LRT). The mixed models allow to account for repeated sessions per patient by
including a random term for subject. The LRT compares the mixed model without the brain image marker
versus the model with the marker. As an effect size, the ΔR² for marginal R²s (i.e. R²s that ignore
random effects in the evaluation) comparing both mixed models is given. A confidence interval for
ΔR² can be computed via bootstrapping, which is computationally expensive and can take up to several
days.

Outputs:
    - for each marker set, the MixedMarkerResult for each brain marker variable is parsed into a
        DataFrame and stored as a CSV in MIXED_MODEL_OUTPUT_DIR
"""

# %%
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from brain_stats_tools.config import (
    MAX_WORKERS,
    MIXED_MODEL_OUTPUT_DIR,
    PREPARED_DATA_DIR,
)
from brain_stats_tools.mixed_model import MixedMarkerResult, fit_marker_mixed_model
from brain_stats_tools.utils import Cols, LongDFCols

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"
N_BOOTSTRAPS_DELTA_R2 = 500


# %%
# list all CSV files with brain markers
csv_files = list(PREPARED_DATA_DIR.glob("*.csv"))

clinical_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

MIXED_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if N_BOOTSTRAPS_DELTA_R2 > 0:
    print(
        "WARNING: Bootstrapping takes several hours up to days."
        "Ensure multithreading and choose a small n (e.g. 500)"
    )
if N_BOOTSTRAPS_DELTA_R2 > 1000:  # noqa: PLR2004
    print(
        f"WARNING: High number of bootstraps ({N_BOOTSTRAPS_DELTA_R2}) will lead to excessive "
        "computation time of several days!"
    )


# %%
# define helper for parallelisation
def run_single_marker(
    marker: str, full_data_df: pd.DataFrame
) -> tuple[str, MixedMarkerResult]:
    """Call function for parallel execution."""
    res = fit_marker_mixed_model(
        df=full_data_df,
        marker_colname=marker,
        target_var_colname=Cols.GMFC,
        n_bootstrap=N_BOOTSTRAPS_DELTA_R2,
    )
    return marker, res


def marker_result_to_dict(marker: str, res: MixedMarkerResult) -> dict:
    """Flatten one MixedMarkerResult into a dict for DataFrame output."""
    return {
        "marker": marker,
        "r2_null": res.r2_null,
        "r2_full": res.r2_full,
        "r2_delta": res.r2_delta,
        "lrt_stat": res.lrt_stat,
        "lrt_df": res.lrt_df,
        "lrt_pvalue": res.lrt_pvalue,
        "beta_marker": res.beta_marker,
        "beta_marker_se": res.beta_marker_se,
        "beta_marker_pvalue": res.beta_marker_pvalue,
        "r2_delta_ci_low": res.r2_delta_ci_low,
        "r2_delta_ci_high": res.r2_delta_ci_high,
        "n_bootstrap": res.n_bootstrap,
    }


# %%
# run analysis with parallelisation of n markers > 1
if __name__ == "__main__":
    for csv in csv_files:
        brain_marker_df = pd.read_csv(csv, sep=";")
        brain_marker_cols = brain_marker_df.columns.tolist()
        brain_marker_cols.remove(LongDFCols.BASENAME)

        full_data_df = brain_marker_df.merge(
            right=clinical_data_df, on=LongDFCols.BASENAME, validate="one_to_one"
        )

        # run models for each marker (parallel if >1 marker)
        results: dict[str, MixedMarkerResult] = {}

        if len(brain_marker_cols) == 1:
            marker = brain_marker_cols[0]
            results[marker] = fit_marker_mixed_model(
                df=full_data_df,
                marker_colname=marker,
                target_var_colname=Cols.GMFC,
                n_bootstrap=N_BOOTSTRAPS_DELTA_R2,
            )
        else:
            # parallel over markers
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # pass the DataFrame into each process
                futures = {
                    executor.submit(run_single_marker, marker, full_data_df): marker
                    for marker in brain_marker_cols
                }
                for fut in as_completed(futures):
                    marker, res = fut.result()
                    results[marker] = res

        # collect into DataFrame and save
        rows = [marker_result_to_dict(marker, res) for marker, res in results.items()]
        result_df = pd.DataFrame(rows)

        marker_set_name = csv.stem.replace("metrics_", "")
        output_name = MIXED_MODEL_OUTPUT_DIR / f"{marker_set_name}_mixed_models.csv"
        result_df.to_csv(output_name, sep=";", index=False)
        print(f"Saved results for {csv.name} -> {output_name}")

# %%
