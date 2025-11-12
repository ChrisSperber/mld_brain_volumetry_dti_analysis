"""Path and configs for MLD MRI statistical analysis."""

from pathlib import Path

ORIGINAL_DATA_ROOT_DIR = Path(__file__).parents[3] / "mld_data"
TEMPORARY_DATA_DIR = Path(__file__).parents[3] / "temp_images"

MP2RAGE = "MP2RAGE"

UNKNOWN = "Unknown"
NOT_APPLICABLE = "Not_applicable"
PATIENT = "patient"
CONTROL = "control"

PROJECTS_DIR = Path(__file__).parents[3]
OUTPUT_METRICS_DIR = PROJECTS_DIR / "MRI_output_metrics"
CH2_EXAMPLE_PIPELINE_DIR = TEMPORARY_DATA_DIR / "ch2_pipeline_demo"

MRI_METRICS_VOLUMETRY_LONG_CSV = (
    OUTPUT_METRICS_DIR / "mri_outcome_metrics_volumetric.csv"
)
MRI_METRICS_FA_LONG_CSV = OUTPUT_METRICS_DIR / "mri_outcome_metrics_FA.csv"
MRI_METRICS_MD_LONG_CSV = OUTPUT_METRICS_DIR / "mri_outcome_metrics_MD.csv"

SUBJECT_EXCLUSION_CSV = Path(__file__).parents[1] / "Exclusions_MLD_segmentation.csv"
