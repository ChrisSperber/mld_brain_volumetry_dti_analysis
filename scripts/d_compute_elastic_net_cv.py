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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, r2_score

from brain_stats_tools.config import (
    MAX_WORKERS,
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
from brain_stats_tools.utils import Cols, LongDFCols

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"

# %%
# list all CSV files with brain markers
csv_files = list(PREPARED_DATA_DIR.glob("*.csv"))

clinical_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# define helper for parallelisation
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
