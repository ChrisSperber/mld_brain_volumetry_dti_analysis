"""Evaluate predictive value of brain imaging markers with elastic net regression.

For each marker set, compute a repeated, cross-validated elastic net regression to assess the
overall predictive value of the entire dataset.

Outputs:
    - for each marker set, the results per cross validation run are stored as a CSV in
        PREDICTION_OUTPUT_DIR; output scores include R², MAE, n/prop of nonzero coefficients,
        top predictor name & coef
"""

# %%

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from brain_stats_tools.config import (
    MAX_WORKERS,
    PREDICTION_OUTPUT_DIR,
    PREPARED_DATA_DIR,
)
from brain_stats_tools.elastic_net_utils import (
    N_PREDICTION_REPS,
    _run_prediction,
)
from brain_stats_tools.utils import Cols, LongDFCols

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"

# %%
# list all CSV files with brain markers
csv_files = list(PREPARED_DATA_DIR.glob("*.csv"))

clinical_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# run analysis with parallelisation between repeated cross validation runs
for csv in csv_files:
    marker_set_name = csv.stem.replace("metrics_", "")
    output_name = PREDICTION_OUTPUT_DIR / f"{marker_set_name}_elastic_net.csv"

    print(f"Starting analysis of {marker_set_name}...")

    # skip if file already exists
    if output_name.exists():
        print(
            f"A prediction output csv for {marker_set_name} already exists at {output_name}"
            " and is skipped."
        )
        continue

    brain_marker_df = pd.read_csv(csv, sep=";")
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
