"""Utility functions and constants for MLD MRI statistical analysis."""

from dataclasses import dataclass


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
