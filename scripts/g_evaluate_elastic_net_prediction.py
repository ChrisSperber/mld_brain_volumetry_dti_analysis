"""Analyse the results of the prediction with elastic net regression.

Outputs:
    - CSV table with various output metrics per condition
    - R² Swarmplots for each condition
    - statistical comparison if numerical top conditions per variable (Volumetry vs. FA vs. MD)
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from brain_stats_tools.config import NOT_AVAILABLE, PREDICTION_OUTPUT_DIR
from brain_stats_tools.utils import (
    PredAnalysisOutputCols,
    analyse_prediction_r2,
    find_unique_path,
    rm_anova_with_posthoc,
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
######################
# create results table

results_list = []
marker_csv_paths: list[Path] = []

for markerset in results_markersets:
    marker_csv_path = find_unique_path(results_csv_files, markerset)
    marker_csv_paths.append(marker_csv_path)
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
##############
# create plots

subplot_titles = [
    "Volume, TIV-adjusted",
    "Fractional Anisotropy, Median",
    "Fractional Anisotropy, p90",
    "Fractional Anisotropy, voxels >0.2",
    "MD, Median",
    "MD, p10",
]

# X-labels per subplot: first only 2 levels, then 3 each
subplot_level_labels = [
    ["Structure-Level", "Region-Level"],  # first subplot (2 CSVs)
    ["Whole-Brain", "Structure-Level", "Region-Level"],  # next 3 CSVs
    ["Whole-Brain", "Structure-Level", "Region-Level"],
    ["Whole-Brain", "Structure-Level", "Region-Level"],
    ["Whole-Brain", "Structure-Level", "Region-Level"],
    ["Whole-Brain", "Structure-Level", "Region-Level"],
]


def load_r2(csv_path: Path, r2_col: str = R2) -> pd.Series:
    """Load R² values from CSV and winsorise to [-1, 1]."""
    df = pd.read_csv(csv_path, sep=";")
    r2 = df[r2_col].astype(float)
    r2 = r2.clip(lower=-1.0, upper=1.0)
    return r2


def make_swarmplots(
    csv_paths: list[Path],
    subplot_titles: list[str],
    subplot_level_labels: list[list[str]],
) -> None:
    """Create 6 subplots of swarmplots with per-level mean bars."""
    # 3x2 grid = 6 axes
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    start_idx = 0
    for subplot_idx, ax in enumerate(axes[:6]):
        # set n marker per plot
        if subplot_idx == 0:
            n_this = 2
        else:
            n_this = 3

        csv_subset = csv_paths[start_idx : start_idx + n_this]
        level_labels = subplot_level_labels[subplot_idx]
        start_idx += n_this

        # Build df: one column for level, one for R²
        frames: list[pd.DataFrame] = []
        for label, csv_path in zip(level_labels, csv_subset, strict=True):
            r2 = load_r2(csv_path)
            frames.append(pd.DataFrame({"level": label, "r2": r2}))

        plot_df = pd.concat(frames, ignore_index=True)

        # Swarmplot of R² per level
        sns.swarmplot(
            data=plot_df,
            x="level",
            y="r2",
            order=level_labels,
            ax=ax,
        )

        # Add mean bar
        means = plot_df.groupby("level")["r2"].mean()
        x_positions = {level: i for i, level in enumerate(level_labels)}
        for level, mean in means.items():
            x = x_positions[level]  # type: ignore
            ax.hlines(
                y=mean, xmin=x - 0.35, xmax=x + 0.35, linewidth=3.5, color="black"
            )

        ax.set_title(subplot_titles[subplot_idx])
        ax.set_xlabel("")
        if subplot_idx % 2 == 0:
            ax.set_ylabel("R²")
        else:
            ax.set_ylabel("")

        # Same scaling across all plots
        ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    outname = Path(__file__).parent / f"{Path(__file__).stem}_r2_plots.png"
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    plt.show()


# call plotting function
make_swarmplots(
    csv_paths=marker_csv_paths,
    subplot_titles=subplot_titles,
    subplot_level_labels=subplot_level_labels,
)

# %%
######################
# statistical analysis
# For accessability, the analysis is limited to the numerically best performing marker sets per
# variable

statistical_analysis_markersets = [
    "volumetry_variable_tiv_level_Region",
    "FA_variable_percent_above_thres_level_Region",
    "MD_variable_p10_level_Region",
]

cols = {}
for markerset in statistical_analysis_markersets:
    csv_path = find_unique_path(results_csv_files, markerset)
    marker_df = pd.read_csv(csv_path, sep=";")
    r2_scores = marker_df[R2].clip(lower=-1, upper=1)
    cols[markerset] = r2_scores

stat_analysis_df = pd.DataFrame(cols)

anova_results = rm_anova_with_posthoc(stat_analysis_df, statistical_analysis_markersets)
print(anova_results.anova_table)
print(anova_results.pairwise)

# write results to txt file
outname = Path(__file__).parent / f"{Path(__file__).stem}_anova_results.txt"
with open(outname, "w", encoding="utf-8") as f:
    # ANOVA table
    f.write("ANOVA results\n")
    f.write("=============\n")
    f.write(anova_results.anova_table.to_string())
    f.write("\n\n")

    # Pairwise tests
    f.write("Post-hoc pairwise comparisons\n")
    f.write("================================\n")
    f.write(anova_results.pairwise.to_string(index=False))
    f.write("\n")

# %%
