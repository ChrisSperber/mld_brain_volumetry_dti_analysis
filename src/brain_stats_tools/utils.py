"""Utility functions and constants for MLD MRI statistical analysis."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brain_stats_tools.config import NOT_APPLICABLE


@dataclass(frozen=True)
class Cols:
    """Data column names."""

    FILENAME: str = "Filename"
    AGE: str = "Age"
    SEX: str = "Sex"
    GMFC: str = "GMFC"
    PATHOLOGY_TYPE: str = "Pathology_Type"
    THERAPY: str = "Therapy"
    MRI_SCORE: str = "MRI_Score"
    SUBJECT_ID: str = "Subject_ID"
    DATE_TAG: str = "Date_Tag"
    IMAGE_MODALITY: str = "Image_Modality"
    DTI_METHOD: str = "DTI_Method"
    SUBJECT_TYPE: str = "Subject_Type"  # Column name for patient/control tag


@dataclass(frozen=True)
class LongDFCols:
    """Data column names for long CSVs."""

    BASENAME: str = "Basename"
    VARIABLE: str = "Variable"
    STRUCTURE: str = "Structure"
    REGION_ID: str = "Region_ID"
    VALUE: str = "Value"


@dataclass(frozen=True)
class AnalysisLevels:
    """Types of analyses levels."""

    REGION: str = "Region"
    STRUCTURE: str = "Structure"
    WHOLE_BRAIN: str = "Whole_Brain"


@dataclass(frozen=True)
class PredAnalysisOutputCols:
    """Types of analyses levels."""

    DATA_MODALITY: str = "Data_Modality"  # volumetry, FA, MD
    VARIABLE: str = "Variable"  # mean, p90 etc.
    DATA_LEVEL: str = "Data_Resolution_Level"  # Whole Brain, Structure, Region
    MEAN_R2: str = "Mean_R2"
    SD_R2: str = "SD_R2"
    R2_STR: str = "R2 Mean_SD"
    MEAN_RATIO_N_NONZERO: str = "Mean_Ratio_Nonzero_Vars"
    TOP_VARIABLES_1: str = "Most_common_top_predictor_1"
    TOP_VARIABLES_2: str = "Most_common_top_predictor_2"
    TOP_VARIABLES_3: str = "Most_common_top_predictor_3"
    TOP_VARIABLES_4: str = "Most_common_top_predictor_4"
    TOP_VARIABLES_5: str = "Most_common_top_predictor_5"


def _split_long_df(df):
    region_df = df[df[LongDFCols.REGION_ID] != NOT_APPLICABLE].copy()

    structure_df = df[
        (df[LongDFCols.REGION_ID] == NOT_APPLICABLE)
        & (df[LongDFCols.STRUCTURE] != NOT_APPLICABLE)
    ].copy()

    whole_brain_df = df[df[LongDFCols.STRUCTURE] == NOT_APPLICABLE].copy()

    return {
        AnalysisLevels.REGION: region_df,
        AnalysisLevels.STRUCTURE: structure_df,
        AnalysisLevels.WHOLE_BRAIN: whole_brain_df,
    }


def analyse_prediction_r2(
    r2_scores: pd.Series, winsorise: bool = True
) -> tuple[float, float]:
    """Analyse a Series of repeated cross validation R² scores.

    Args:
        r2_scores: Series of R² values.
        winsorise: Winsorises extreme negative values <-1 if True.

    Returns:
        tuple[float,float]: tuple containing mean and sd of R².

    """
    if np.any(r2_scores > 1):
        raise ValueError("R² values > 1 detected. Check your pipeline.")
    if winsorise:
        r2_scores = r2_scores.clip(lower=-1, upper=1)

    mean_r2 = r2_scores.mean()
    std_r2 = r2_scores.std()

    return mean_r2, std_r2


def find_unique_path(paths: list[Path], substring: str) -> Path:
    """Find a unique filepath in a list containing a unique string.

    Args:
        paths: List of path objects.
        substring: Substring to match

    Raises:
        ValueError: More than 1 matching file exists.

    Returns:
        Unique file path.

    """
    matches = [p for p in paths if substring in str(p)]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Multiple matches found for {substring}")
    else:
        raise ValueError(f"No matches found for {substring}")
