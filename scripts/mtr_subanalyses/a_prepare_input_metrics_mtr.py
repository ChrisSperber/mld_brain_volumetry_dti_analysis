"""Collect MTR data from long CSVs generated in the MRI processing repo.

Requirements:
    - MTR data were processed with the metachromatic_leukodystrophy_mri_processing repo
        pipeline, generating long CSVs with metrics derived from segmentation and DTI

Outputs:
    - wide CSVs with relevant data for included subjects are created for MTR for each analysis
        level (region-wise/structure-wise/whole brain) in PREPARED_DATA_DIR
"""

# %%
import pandas as pd

from brain_stats_tools.config import (
    MRI_METRICS_MTR_LONG_CSV,
    PREPARED_DATA_DIR_MTR,
    SUBJECT_EXCLUSION_CSV,
)
from brain_stats_tools.utils import Cols, LongDFCols, _split_long_df

VARIABLES_MTR = ["median", "p10"]

SUBSTRINGS_TO_DROP_FROM_COLNAMES = [
    "_median_mtr",
    "_p10_mtr",
]

# %%
# read CSV
mri_metrics_mtr_long_df = pd.read_csv(MRI_METRICS_MTR_LONG_CSV, sep=";")

exclusions_df = pd.read_csv(SUBJECT_EXCLUSION_CSV, sep=";")
exclusions_df[LongDFCols.BASENAME] = (
    "subject_"
    + exclusions_df[Cols.SUBJECT_ID].astype(str)
    + "_date_"
    + exclusions_df[Cols.DATE_TAG].astype(str)
)
to_exclude = set(exclusions_df[LongDFCols.BASENAME])

# drop excluded subjects
for long_df in [
    mri_metrics_mtr_long_df,
]:
    long_df.drop(
        index=long_df.index[long_df[LongDFCols.BASENAME].isin(to_exclude)],
        inplace=True,
    )

# replace - (minus) from variable names and with underscore
mri_metrics_mtr_long_df[LongDFCols.VARIABLE] = mri_metrics_mtr_long_df[
    LongDFCols.VARIABLE
].str.replace("-", "_", regex=False)

# %%
# split into data types (region/structure/whole_brain)
mri_metrics_mtr_long_dfs_dict = _split_long_df(mri_metrics_mtr_long_df)

# cleanup to prevent issues with VSCode viewer extensions
del mri_metrics_mtr_long_df

# %%
# convert to wide tables and store locally
PREPARED_DATA_DIR_MTR.mkdir(exist_ok=True, parents=True)

# MD
for analysis_level in mri_metrics_mtr_long_dfs_dict.keys():
    for variable in VARIABLES_MTR:
        long_df = mri_metrics_mtr_long_dfs_dict[analysis_level]
        long_df = long_df[
            long_df[LongDFCols.VARIABLE].str.contains(variable, na=False)
        ].copy()

        wide_df = long_df.pivot(
            index=LongDFCols.BASENAME,
            columns=LongDFCols.VARIABLE,
            values=LongDFCols.VALUE,
        )

        for item in SUBSTRINGS_TO_DROP_FROM_COLNAMES:
            wide_df.columns = wide_df.columns.str.replace(item, "", regex=False)

        outputname = f"metrics_MTR_variable_{variable}_level_{analysis_level}.csv"
        outpath = PREPARED_DATA_DIR_MTR / outputname
        wide_df.to_csv(outpath, sep=";")

# %%
