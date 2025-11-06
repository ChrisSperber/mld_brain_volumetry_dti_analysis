"""Path and configs for MLD MRI statistical analysis."""

from pathlib import Path

ORIGINAL_DATA_ROOT_DIR = Path(__file__).parents[3] / "mld_data"
TEMPORARY_DATA_DIR = Path(__file__).parents[3] / "temp_images"

MP2RAGE = "MP2RAGE"

UNKNOWN = "Unknown"
NOT_APPLICABLE = "Not_applicable"
PATIENT = "patient"
CONTROL = "control"

MRI_PROCESSING_DIR = (
    Path(__file__).parents[3] / "metachromatic_leukodystrophy_mri_processing"
)
OUTPUT_METRICS_DIR = MRI_PROCESSING_DIR / "MRI_output_metrics"
