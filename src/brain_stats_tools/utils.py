"""Utility functions and constants for MLD MRI statistical analysis."""

from dataclasses import dataclass

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
