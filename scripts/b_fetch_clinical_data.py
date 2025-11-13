"""Fetch clinical data from mri_processing repo and clean/format/store in local repo.

Requirements:
    - metachromatic_leukodystrophy_mri_processing repo outputs are available

Outputs:
    - CSV with relevant (anonymised) clinical and demographic data
"""

# %%
from pathlib import Path

import pandas as pd

from brain_stats_tools.config import CLINICAL_DATA_CSV, SUBJECT_EXCLUSION_CSV
from brain_stats_tools.utils import Cols, LongDFCols

FA = "FA"

# %%

original_data_df = pd.read_csv(CLINICAL_DATA_CSV, sep=";")
original_data_df = original_data_df[original_data_df[Cols.IMAGE_MODALITY] == FA]

cols_to_drop = [
    Cols.IMAGE_MODALITY,
    Cols.FILENAME,
    "Image_Shape",
    "Image_Dtype",
]
original_data_df = original_data_df.drop(cols_to_drop, axis=1)
original_data_df[LongDFCols.BASENAME] = (
    "subject_"
    + original_data_df[Cols.SUBJECT_ID].astype(str)
    + "_date_"
    + original_data_df[Cols.DATE_TAG].astype(str)
)


exclusions_df = pd.read_csv(SUBJECT_EXCLUSION_CSV, sep=";")
exclusions_df[LongDFCols.BASENAME] = (
    "subject_"
    + exclusions_df[Cols.SUBJECT_ID].astype(str)
    + "_date_"
    + exclusions_df[Cols.DATE_TAG].astype(str)
)
to_exclude = set(exclusions_df[LongDFCols.BASENAME])

original_data_df.drop(
    index=original_data_df.index[
        original_data_df[LongDFCols.BASENAME].isin(to_exclude)
    ],
    inplace=True,
)

# %%
output_name = Path(__file__).with_suffix(".csv")
original_data_df.to_csv(output_name, index=False, sep=";")

# %%
