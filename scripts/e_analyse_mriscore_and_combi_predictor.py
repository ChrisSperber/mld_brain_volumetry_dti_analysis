"""Additional prediction/mixed model analyses.

This script analyses the MRIscore (a visual rating score of the MRI) in mixed models/prediction and
a combi concatenation model of both FA (% voxel above threshold) and volumetry in prediction.
Methods and output formats mirror the main analyses on volumery/FA/MD. Random seeds are picked
equivalently to ensure that procedures are replicated.

Outputs:
    - for prediction, the results per cross validation run are stored as a CSV in
        PREDICTION_OUTPUT_DIR; output scores include R², MAE, n/prop of nonzero coefficients,
        top predictor name & coef
    - for mixed model analysis, the MixedMarkerResult for each brain marker variable is parsed into
        a DataFrame and stored as a CSV in MIXED_MODEL_OUTPUT_DIR
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, r2_score

from brain_stats_tools.config import (
    MAX_WORKERS,
    MIXED_MODEL_OUTPUT_DIR,
    PREDICTION_OUTPUT_DIR,
    PREPARED_DATA_DIR,
)
from brain_stats_tools.elastic_net_utils import (
    N_PREDICTION_REPS,
    NONZERO_COEFF_THRESHOLD,
    TEST_SIZE_RATIO,
    fit_elastic_net_bayes_opt,
    train_test_split_indices,
)
from brain_stats_tools.mixed_model import (
    N_BOOTSTRAPS_DELTA_R2,
    MixedMarkerResult,
    fit_marker_mixed_model,
)
from brain_stats_tools.utils import Cols, LongDFCols, find_unique_path

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"

VOLUMETRY_CSV_SUBSTRING = "volumetry_variable_tiv_level_Region"
FA_CSV_SUBSTRING = "FA_variable_percent_above_thres_level_Region"


# %%
# list all CSV files with brain markers
csv_files = list(PREPARED_DATA_DIR.glob("*.csv"))

clinical_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

# %%
# load and combine FA/volumetry data
volumetry_csv = find_unique_path(csv_files, VOLUMETRY_CSV_SUBSTRING)
volumetry_df = pd.read_csv(volumetry_csv, sep=";")
fa_csv = find_unique_path(csv_files, FA_CSV_SUBSTRING)
fa_df = pd.read_csv(fa_csv, sep=";")

merged_markers_df = pd.merge(
    left=volumetry_df, right=fa_df, on=LongDFCols.BASENAME, validate="one_to_one"
)

# %%
# create df with MRIscore mirroring the prepared MRI marker CSVs' structure
mri_score_pred_df = clinical_data_df[[LongDFCols.BASENAME, Cols.MRI_SCORE]]


# %%
# define helper for parallelisation - prediction
def _run_prediction(  # noqa: PLR0913
    split_id: int,
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    marker_set_name: str,
    predictor_names: list[str],
) -> dict[str, object]:
    train_idx, test_idx = train_test_split_indices(
        n=n_samples,
        test_ratio=TEST_SIZE_RATIO,
        seed=split_id,
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit model on training data
    model = fit_elastic_net_bayes_opt(x_train, y_train)

    # Test predictions
    y_pred = model.predict(x_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Extract coefficients from the ElasticNet step inside the pipeline
    elastic = model.named_steps[  # pyright: ignore[reportAttributeAccessIssue]
        "elasticnet"
    ]
    coefs = elastic.coef_  # shape: (n_features,)

    # Sanity check: coefficients should align with predictor_names
    assert coefs.shape[0] == len(predictor_names)  # noqa: S101

    abs_coefs = np.abs(coefs)

    # Non-zero coefficients according to threshold
    nonzero_mask = abs_coefs >= NONZERO_COEFF_THRESHOLD
    n_nonzero = int(nonzero_mask.sum())
    prop_nonzero = float(n_nonzero / coefs.shape[0])

    # Variable with highest absolute coefficient
    top_idx = int(abs_coefs.argmax())
    top_variable = predictor_names[top_idx]
    top_coef = float(coefs[top_idx])

    return {
        "marker_set": marker_set_name,
        "split": split_id,
        "r2": r2,
        "mae": mae,
        "n_nonzero": n_nonzero,
        "prop_nonzero": prop_nonzero,
        "top_variable": top_variable,
        "top_coef": top_coef,
    }


# %%
# define helper for parallelisation - mixed models
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
# prediction elastic net - combined model

marker_set_name = "volumetryFA_variable_combined_level_na"
output_name = PREDICTION_OUTPUT_DIR / f"{marker_set_name}_elastic_net.csv"

if output_name.exists():
    print(
        f"A prediction output csv for {marker_set_name} already exists at {output_name}"
        " and is skipped."
    )
else:
    print(f"Starting analysis of {marker_set_name}...")

    brain_marker_df = merged_markers_df.copy()
    brain_marker_cols = brain_marker_df.columns.tolist()
    brain_marker_cols.remove(LongDFCols.BASENAME)

    full_data_df = brain_marker_df.merge(
        right=clinical_data_df,
        on=LongDFCols.BASENAME,
        validate="one_to_one",
    )

    X = full_data_df[brain_marker_cols].to_numpy(dtype=float)
    y = full_data_df[Cols.GMFC].to_numpy()
    n_samples = X.shape[0]

    # perform parallelised analysis
    results = Parallel(n_jobs=MAX_WORKERS, verbose=10)(
        delayed(_run_prediction)(
            split_id,
            X,
            y,
            n_samples,
            marker_set_name,
            brain_marker_cols,
        )
        for split_id in range(N_PREDICTION_REPS)
    )

    results_df = pd.DataFrame(results)  # pyright: ignore[reportArgumentType]
    results_df.to_csv(output_name, sep=";", index=False)
    print(f"Finished {marker_set_name}, saved to {output_name}")

# %%
# prediction elastic net - MRIscore
marker_set_name = "MRIscore_variable_na_level_na"
output_name = PREDICTION_OUTPUT_DIR / f"{marker_set_name}_elastic_net.csv"

if output_name.exists():
    print(
        f"A prediction output csv for {marker_set_name} already exists at {output_name}"
        " and is skipped."
    )
else:
    print(f"Starting analysis of {marker_set_name}...")

    brain_marker_df = clinical_data_df[[LongDFCols.BASENAME, Cols.MRI_SCORE]].copy()
    brain_marker_cols = brain_marker_df.columns.tolist()
    brain_marker_cols.remove(LongDFCols.BASENAME)

    full_data_df = brain_marker_df.merge(
        right=clinical_data_df,
        on=[LongDFCols.BASENAME, Cols.MRI_SCORE],
        validate="one_to_one",
    )

    X = full_data_df[brain_marker_cols].to_numpy(dtype=float)
    y = full_data_df[Cols.GMFC].to_numpy()
    n_samples = X.shape[0]

    # perform parallelised analysis
    results = Parallel(n_jobs=MAX_WORKERS, verbose=10)(
        delayed(_run_prediction)(
            split_id,
            X,
            y,
            n_samples,
            marker_set_name,
            brain_marker_cols,
        )
        for split_id in range(N_PREDICTION_REPS)
    )

    results_df = pd.DataFrame(results)  # pyright: ignore[reportArgumentType]
    results_df.to_csv(output_name, sep=";", index=False)
    print(f"Finished {marker_set_name}, saved to {output_name}")

# %%
# mixed model analysis - MRIscore
marker_set_name = "MRIscore_variable_na_level_na"
output_name = MIXED_MODEL_OUTPUT_DIR / f"{marker_set_name}_mixed_models.csv"

if output_name.exists():
    print(
        f"A mixed model csv for {marker_set_name} already exists at {output_name}"
        " and is skipped."
    )
else:
    brain_marker_df = clinical_data_df[[LongDFCols.BASENAME, Cols.MRI_SCORE]].copy()
    brain_marker_cols = brain_marker_df.columns.tolist()
    brain_marker_cols.remove(LongDFCols.BASENAME)
    full_data_df = brain_marker_df.merge(
        right=clinical_data_df,
        on=[LongDFCols.BASENAME, Cols.MRI_SCORE],
        validate="one_to_one",
    )

    # run models for each marker
    results_mixed_model: dict[str, MixedMarkerResult] = {}

    marker = brain_marker_cols[0]
    results_mixed_model[marker] = fit_marker_mixed_model(
        df=full_data_df,
        marker_colname=marker,
        target_var_colname=Cols.GMFC,
        n_bootstrap=N_BOOTSTRAPS_DELTA_R2,
    )

    # collect into DataFrame and save
    rows = [
        marker_result_to_dict(marker, res)
        for marker, res in results_mixed_model.items()
    ]
    result_df = pd.DataFrame(rows)

    result_df.to_csv(output_name, sep=";", index=False)
    print(f"Saved results for {marker_set_name} -> {output_name}")

print("Done!")

# %%
