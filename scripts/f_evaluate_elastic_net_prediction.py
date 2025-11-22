"""Analyse the results of the prediction with elastic net regression.

Outputs:
    - CSV table with various output metrics per condition
    - xxx
"""

# %%
from pathlib import Path

import pandas as pd

from brain_stats_tools.config import NOT_AVAILABLE, PREDICTION_OUTPUT_DIR
from brain_stats_tools.utils import (
    PredAnalysisOutputCols,
    analyse_prediction_r2,
    find_unique_path,
)

# column names in the results CSVs
R2 = "r2"
N_NONZERO = "n_nonzero"
PROP_NONZERO = "prop_nonzero"
TOP_VARIABLE = "top_variable"

results_markersets = [
    "volumetry_variable_tiv_level_Structure",
    "volumetry_variable_tiv_level_Region",
    "FA_variable_median_level_Whole_Brain",
    "FA_variable_median_level_Structure",
    "FA_variable_median_level_Region",
    "FA_variable_p90_level_Whole_Brain",
    "FA_variable_p90_level_Structure",
    "FA_variable_p90_level_Region",
    "FA_variable_percent_above_thres_level_Whole_Brain",
    "FA_variable_percent_above_thres_level_Structure",
    "FA_variable_percent_above_thres_level_Region",
    "MD_variable_median_level_Whole_Brain",
    "MD_variable_median_level_Structure",
    "MD_variable_median_level_Region",
    "MD_variable_p10_level_Whole_Brain",
    "MD_variable_p10_level_Structure",
    "MD_variable_p10_level_Region",
]


# %%
# list all CSV files with brain markers
results_csv_files = list(PREDICTION_OUTPUT_DIR.glob("*.csv"))

# %%
# create results table
results_list = []

for markerset in results_markersets:
    marker_csv_path = find_unique_path(results_csv_files, markerset)
    marker_df = pd.read_csv(marker_csv_path, sep=";")
    mean_r2, sd_r2 = analyse_prediction_r2(marker_df[R2])
    r2_str = f"{mean_r2} ± {sd_r2}"

    data_modality = markerset.split("_")[0]
    _, var_level_str = markerset.split("variable_", 1)
    variable, level = var_level_str.split("_level_", 1)

    nonzero = marker_df.loc[0, N_NONZERO]
    prop = marker_df.loc[0, PROP_NONZERO]
    n_coeffs_total = int(round(nonzero / prop))  # type: ignore
    mean_n_coeffs = round(marker_df[N_NONZERO].mean(), 1)
    n_coeffs_str = f"{mean_n_coeffs}/{n_coeffs_total}"

    top_variables = marker_df[TOP_VARIABLE]
    top_variables = top_variables.str.replace("_structure", "", regex=False)
    counts = top_variables.value_counts()
    top5 = counts.head(5)

    # create strings of top 5 variables
    top_variable_strings = []
    for var, count in top5.items():
        percentage = (count / len(top_variables)) * 100
        formatted_string = f"{var}_{percentage:.1f}%"
        top_variable_strings.append(formatted_string)
    while len(top_variable_strings) < 5:  # noqa: PLR2004
        top_variable_strings.append(NOT_AVAILABLE)

    results_list.append(
        {
            PredAnalysisOutputCols.DATA_MODALITY: data_modality,
            PredAnalysisOutputCols.VARIABLE: variable,
            PredAnalysisOutputCols.DATA_LEVEL: level,
            PredAnalysisOutputCols.MEAN_R2: mean_r2,
            PredAnalysisOutputCols.SD_R2: sd_r2,
            PredAnalysisOutputCols.R2_STR: r2_str,
            PredAnalysisOutputCols.MEAN_RATIO_N_NONZERO: n_coeffs_str,
            PredAnalysisOutputCols.TOP_VARIABLES_1: top_variable_strings[0],
            PredAnalysisOutputCols.TOP_VARIABLES_2: top_variable_strings[1],
            PredAnalysisOutputCols.TOP_VARIABLES_3: top_variable_strings[2],
            PredAnalysisOutputCols.TOP_VARIABLES_4: top_variable_strings[3],
            PredAnalysisOutputCols.TOP_VARIABLES_5: top_variable_strings[4],
        }
    )

# %%
# store results table
results_df = pd.DataFrame(results_list)
output_name = Path(__file__).with_suffix(".csv")
results_df.to_csv(output_name, index=False, sep=";")

# %%
# create plots
pass

# %%
# statistical analysis
pass

# %%
