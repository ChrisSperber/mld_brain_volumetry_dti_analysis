"""Utility functions and constants for MLD MRI statistical analysis."""

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

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

    mean_r2 = round(mean_r2, 3)
    std_r2 = round(std_r2, 3)
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


@dataclass
class RmAnovaResult:
    """Results of a one-way repeated-measures ANOVA with post-hoc tests."""

    anova_table: pd.DataFrame  # statsmodels AnovaRM table
    pairwise: pd.DataFrame  # one row per pairwise comparison


def rm_anova_with_posthoc(
    df: pd.DataFrame,
    cond_cols: list[str],
) -> RmAnovaResult:
    """Run one-way repeated-measures ANOVA and Bonferroni-corrected t-tests.

    Args:
        df: wide-format DataFrame, each row = subject, each cond_col = condition.
        cond_cols: names of the repeated-measures condition columns.

    Returns:
        RmAnovaResult with ANOVA table, partial eta², and pairwise results.

    """
    df = df.copy()
    df["subject"] = np.arange(len(df))

    # Long format for AnovaRM
    long_df = df.melt(
        id_vars="subject",
        value_vars=cond_cols,
        var_name="condition",
        value_name="score",
    )

    # Repeated-measures ANOVA
    anova = AnovaRM(
        data=long_df,
        depvar="score",
        subject="subject",
        within=["condition"],
    ).fit()
    anova_table = anova.anova_table

    # Pairwise post-hoc tests (paired t-tests, Bonferroni corrected)
    pairs = list(combinations(cond_cols, 2))
    m = len(pairs)  # number of comparisons

    rows = []
    for c1, c2 in pairs:
        pair_df = df[[c1, c2]].dropna()
        x = pair_df[c1].to_numpy()
        y = pair_df[c2].to_numpy()

        t_stat, p_raw = ttest_rel(x, y)
        p_bonf = min(p_raw * m, 1.0)
        n = len(pair_df)
        d_z = t_stat / sqrt(n) if n > 0 else np.nan

        rows.append(
            {
                "cond1": c1,
                "cond2": c2,
                "n": n,
                "t": float(t_stat),
                "p_raw": float(p_raw),
                "p_bonf": float(p_bonf),
                "cohens_dz": float(d_z),
            }
        )

    pairwise_df = pd.DataFrame(rows)

    return RmAnovaResult(
        anova_table=anova_table,
        pairwise=pairwise_df,
    )


def reassign_consecutive_labels(input_array: np.ndarray) -> np.ndarray:
    """Reassigns consecutive integer labels starting from 1, keeping 0 as it is.

    Args:
        input_array (np.ndarray): An array containing the original labels.

    Returns:
        np.ndarray: A new array with reassigned consecutive integer labels, keeping 0.

    """
    unique_labels = np.unique(input_array)
    unique_labels_no_zero = unique_labels[unique_labels != 0]
    label_mapping = {label: idx + 1 for idx, label in enumerate(unique_labels_no_zero)}
    output_array = np.vectorize(lambda x: label_mapping.get(x, 0))(input_array)

    return output_array


def load_nifti(path: str | Path) -> Nifti1Image:
    """Load NIFTI Wrapper with typechecking to silence Pylance warnings."""
    img = nib.load(str(path))  # pyright: ignore[reportPrivateImportUsage]
    if not isinstance(img, Nifti1Image):
        raise TypeError("Unexpected image type")
    return img


def save_nifti(nifti: Nifti1Image, outname: Path) -> None:
    """Save NIFTI Wrapper to silence Pylance warnings.

    Args:
        nifti: Nifti image.
        outname: Out path.

    """
    nib.save(nifti, outname)  # pyright: ignore[reportPrivateImportUsage]
