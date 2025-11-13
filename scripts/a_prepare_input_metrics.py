"""Collect all relevant data from long CSVs generated in the MRI processing repo.

Requirements:
    - imaging data were processed with the metachromatic_leukodystrophy_mri_processing repo
        pipeline, generating long CSVs with metrics derived from segmentation and DTI
    - all segmentations underwent visual QC and exclusions are documented in a manually generated
        CSV under brain_stats_tools.config SUBJECT_EXCLUSION_CSV

Outputs:
    - wide CSVs with relevant data for included subjects are created for each variable and analysis
        level (region-wise/structure-wise/whole brain) in PREPARED_DATA_DIR
"""

# %%
import pandas as pd

from brain_stats_tools.config import (
    MRI_METRICS_FA_LONG_CSV,
    MRI_METRICS_MD_LONG_CSV,
    MRI_METRICS_VOLUMETRY_LONG_CSV,
    PREPARED_DATA_DIR,
    SUBJECT_EXCLUSION_CSV,
)
from brain_stats_tools.utils import AnalysisLevels, Cols, LongDFCols, _split_long_df

VARIABLES_FA = ["median", "p90", "percent_above_thres"]
VARIABLES_MD = ["median", "p10"]
VARIABLES_VOLUMETRY = ["tiv"]

SUBSTRINGS_TO_DROP_FROM_COLNAMES = [
    "_median_fa",
    "_median_md",
    "_percent_tiv",
    "_p90_fa",
    "_p10_md",
    "percent_above_thres_fa",
]

# %%
# read CSVs
mri_metrics_fa_long_df = pd.read_csv(MRI_METRICS_FA_LONG_CSV, sep=";")
mri_metrics_md_long_df = pd.read_csv(MRI_METRICS_MD_LONG_CSV, sep=";")
mri_metrics_volumetry_long_df = pd.read_csv(MRI_METRICS_VOLUMETRY_LONG_CSV, sep=";")

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
    mri_metrics_fa_long_df,
    mri_metrics_md_long_df,
    mri_metrics_volumetry_long_df,
]:
    long_df.drop(
        index=long_df.index[long_df[LongDFCols.BASENAME].isin(to_exclude)],
        inplace=True,
    )

# %%
# split into data types (region/structure/whole_brain)
mri_metrics_fa_long_dfs_dict = _split_long_df(mri_metrics_fa_long_df)
mri_metrics_md_long_dfs_dict = _split_long_df(mri_metrics_md_long_df)
mri_metrics_volumetry_long_dfs_dict = _split_long_df(mri_metrics_volumetry_long_df)

# cleanup to prevent issues with VSCode viewer extensions
del mri_metrics_fa_long_df, mri_metrics_md_long_df, mri_metrics_volumetry_long_df

# %%
# convert to wide tables adn store locally
PREPARED_DATA_DIR.mkdir(exist_ok=True, parents=True)

# FA
for analysis_level in mri_metrics_fa_long_dfs_dict.keys():
    for variable in VARIABLES_FA:
        long_df = mri_metrics_fa_long_dfs_dict[analysis_level]
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

        outputname = f"metrics_FA_variable_{variable}_level_{analysis_level}.csv"
        outpath = PREPARED_DATA_DIR / outputname
        wide_df.to_csv(outpath, sep=";")

# MD
for analysis_level in mri_metrics_md_long_dfs_dict.keys():
    for variable in VARIABLES_MD:
        long_df = mri_metrics_md_long_dfs_dict[analysis_level]
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

        outputname = f"metrics_MD_variable_{variable}_level_{analysis_level}.csv"
        outpath = PREPARED_DATA_DIR / outputname
        wide_df.to_csv(outpath, sep=";")

# VOLUMETRY
for analysis_level in mri_metrics_volumetry_long_dfs_dict.keys():
    # no whole brain volumetry here, hence skip
    if analysis_level == AnalysisLevels.WHOLE_BRAIN:
        continue

    for variable in VARIABLES_VOLUMETRY:
        long_df = mri_metrics_volumetry_long_dfs_dict[analysis_level]
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

        outputname = f"metrics_volumetry_variable_{variable}_level_{analysis_level}.csv"
        outpath = PREPARED_DATA_DIR / outputname
        wide_df.to_csv(outpath, sep=";")

# %%
