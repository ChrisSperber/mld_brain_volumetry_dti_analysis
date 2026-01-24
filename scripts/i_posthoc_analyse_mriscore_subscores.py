"""Additional prediction/mixed model analyses on MRI score subscores.

This script analyses the subratings of the MRIscore (a visual rating score of the MRI) in mixed
models/prediction. Methods and output formats mirror the main analyses on volumetry/FA/MD.
This analysis is only run on patients, as controls did not receive a rating (and likely would just
present a general ceiling effect across all variables).

Requirements:
    - the output of metachromatic_leukodystrophy_mri_processing
        scripts/mri_scores_processing/a_collect_mri_scores.csv
        is locally available at MRI_SCORES_CSV

Outputs:
    - for prediction, the results per cross validation run are stored as a CSV in
        PREDICTION_OUTPUT_DIR; output scores include R², MAE, n/prop of nonzero coefficients,
        top predictor name & coef
    - for mixed model analysis, the MixedMarkerResult for each brain marker variable is parsed into
        a DataFrame and stored as a CSV in MIXED_MODEL_OUTPUT_DIR
"""

# %%
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from brain_stats_tools.config import (
    MAX_WORKERS,
    MIXED_MODEL_OUTPUT_DIR,
    PREDICTION_OUTPUT_DIR,
)
from brain_stats_tools.elastic_net_utils import (
    N_PREDICTION_REPS,
    _run_prediction,
)
from brain_stats_tools.mixed_model import (
    MixedMarkerResult,
    marker_result_to_dict,
    run_single_marker,
)
from brain_stats_tools.utils import Cols, LongDFCols, MRIScoreSubmarker

MRI_SCORES_CSV = (
    Path(__file__).parents[2]
    / "metachromatic_leukodystrophy_mri_processing"
    / "scripts"
    / "mri_scores_processing"
    / "a_collect_mri_scores.csv"
)

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"

# %%
# load data
data_df = pd.read_csv(MRI_SCORES_CSV, sep=";")
clinical_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

# %%
# prediction elastic net
marker_set_name = "MRIscore_variable_subscores_level_na"
output_name = PREDICTION_OUTPUT_DIR / f"{marker_set_name}_elastic_net.csv"

if output_name.exists():
    print(
        f"A prediction output csv for {marker_set_name} already exists at {output_name}"
        " and is skipped."
    )
else:
    print(f"Starting analysis of {marker_set_name}...")

    brain_marker_df = data_df.copy()
    brain_marker_cols = MRIScoreSubmarker

    X = data_df[brain_marker_cols].to_numpy(dtype=float)
    y = data_df[Cols.GMFC].to_numpy()
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
marker_set_name = "MRIscore_variable_subscores_level_na"
output_name = MIXED_MODEL_OUTPUT_DIR / f"{marker_set_name}_mixed_models.csv"


if output_name.exists():
    print(
        f"A mixed model csv for {marker_set_name} already exists at {output_name}"
        " and is skipped."
    )
else:
    brain_marker_df = data_df[[LongDFCols.BASENAME, *MRIScoreSubmarker]].copy()
    brain_marker_cols = brain_marker_df.columns.tolist()
    brain_marker_cols.remove(LongDFCols.BASENAME)
    full_data_df = brain_marker_df.merge(
        right=clinical_data_df,
        on=[LongDFCols.BASENAME],
        validate="one_to_one",
    )

    # identify markers with variance
    marker_var = brain_marker_df[brain_marker_cols].var(axis=0, skipna=True)

    valid_markers = marker_var[marker_var > 0].index.tolist()
    dropped_markers = set(brain_marker_cols) - set(valid_markers)

    if dropped_markers:
        print(f"Dropping {len(dropped_markers)} zero-variance markers")

    brain_marker_cols = valid_markers

    # run models for each marker
    results_mixed_model: dict[str, MixedMarkerResult] = {}

    # parallel over markers
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # pass the DataFrame into each process
        futures = {
            executor.submit(run_single_marker, marker, full_data_df): marker
            for marker in brain_marker_cols
        }
        for fut in as_completed(futures):
            marker, res = fut.result()
            results_mixed_model[marker] = res

    # collect into DataFrame and save
    rows = [
        marker_result_to_dict(marker, res)
        for marker, res in results_mixed_model.items()
    ]
    result_df = pd.DataFrame(rows)

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
