"""Path and configs for MLD MRI statistical analysis."""

from pathlib import Path

RNG_SEED = 9001
MAX_WORKERS = 6

PROJECTS_DIR = Path(__file__).parents[3]
ORIGINAL_DATA_ROOT_DIR = PROJECTS_DIR / "mld_data"
TEMPORARY_DATA_DIR = PROJECTS_DIR / "temp_images"

MP2RAGE = "MP2RAGE"

UNKNOWN = "Unknown"
NOT_APPLICABLE = "Not_applicable"
PATIENT = "patient"
CONTROL = "control"

NOT_AVAILABLE = "Not_available"

OUTPUT_METRICS_DIR = PROJECTS_DIR / "mld_MRI_output_metrics"
CH2_EXAMPLE_PIPELINE_DIR = TEMPORARY_DATA_DIR / "ch2_pipeline_demo"

MRI_METRICS_VOLUMETRY_LONG_CSV = (
    OUTPUT_METRICS_DIR / "mri_outcome_metrics_volumetric.csv"
)
MRI_METRICS_FA_LONG_CSV = OUTPUT_METRICS_DIR / "mri_outcome_metrics_FA.csv"
MRI_METRICS_MD_LONG_CSV = OUTPUT_METRICS_DIR / "mri_outcome_metrics_MD.csv"
MRI_METRICS_MTR_LONG_CSV = OUTPUT_METRICS_DIR / "mri_outcome_metrics_MTR.csv"

SUBJECT_EXCLUSION_CSV = Path(__file__).parents[1] / "Exclusions_MLD_segmentation.csv"

PREPARED_DATA_DIR = Path(__file__).parents[2] / "prepared_input_data"
PREPARED_DATA_DIR_MTR = Path(__file__).parents[2] / "prepared_input_data_mtr"
MIXED_MODEL_OUTPUT_DIR = Path(__file__).parents[2] / "mixed_models_results"
PREDICTION_OUTPUT_DIR = Path(__file__).parents[2] / "prediction_results"
PREDICTION_OUTPUT_DIR_MTR = Path(__file__).parents[2] / "prediction_results_mtr"

CLINICAL_DATA_CSV = (
    PROJECTS_DIR
    / "metachromatic_leukodystrophy_mri_processing"
    / "scripts"
    / "b_collect_and_verify_data.csv"
)
SEGMENTATION_LABELS_CSV = (
    PROJECTS_DIR
    / "metachromatic_leukodystrophy_mri_processing"
    / "scripts"
    / "g_fetch_freesurfer_labelmap.csv"
)

SEGMENTATION_VISUALISATION_NIFTI_DIR = (
    Path(__file__).parents[2] / "segmentation_visualisation_images"
)
