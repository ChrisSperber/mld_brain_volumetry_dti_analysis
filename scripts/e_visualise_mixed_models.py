"""Visualise statistical mapping results on ch2 template.

The visualisation was limited to variables that allowed the best prediction performances in the
elastic net regression, i.e. volumetry, MD p10, FA percent voxels above threshold.

Requirements:
    - The ch2 T1 MRI template was parcellated with the code in the
        metachromatic_leukodystrophy_mri_processing repository and stored to
        CH2_EXAMPLE_PIPELINE_DIR
    - The outputs of the metachromatic_leukodystrophy_mri_processing repository are locally present
        and SEGMENTATION_LABELS_CSV points to the csv containing the label map
        g_fetch_freesurfer_labelmap.csv

Outputs:
    - aesthetically modified ch2 parcellation/voronoi images in NIFTI format for method
        visualisation to be used e.g. in MRIcron
    - NIFTI images of the mixed models results, mapping the image marker effect size
        (i.e. Δ marginal R²) for all markers that are significant after Bonferroni correction
        (based on the statistical results of the likelihood ratio test) onto the ch2 template
"""

# %%

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image

from brain_stats_tools.config import (
    CH2_EXAMPLE_PIPELINE_DIR,
    MIXED_MODEL_OUTPUT_DIR,
    SEGMENTATION_LABELS_CSV,
    SEGMENTATION_VISUALISATION_NIFTI_DIR,
)
from brain_stats_tools.utils import (
    load_nifti,
    reassign_consecutive_labels,
    save_nifti,
)

ALHPA_LEVEL = (
    0.05  # alpha p-level for significance, applied to Bonferroni corrected p values
)

CH2_SNYTHSEG_PARCELLATION = CH2_EXAMPLE_PIPELINE_DIR / "ch2_synthseg_labels.nii.gz"
CH2_WM_VORONOI_PARCELLATION = CH2_EXAMPLE_PIPELINE_DIR / "ch2_WM_voronoi_labels.nii.gz"

CH2_EXAMPLE_SEGMENTATION_DIR = SEGMENTATION_VISUALISATION_NIFTI_DIR / "ch2_examples"
MAPPING_VISUALISATION_OUT_DIR = SEGMENTATION_VISUALISATION_NIFTI_DIR / "mapping_results"

STRUCTURE = "Structure"
STRUCTURE_ID = "Structure_ID"
REGION_ID = "id"
LABEL = "Label"
LABEL_UNDERSCORE = "Labels_Underscore"

R2_DELTA = "r2_delta"
R2_DELTA_SIGNIFICANT = "r2_delta_significant"
P_BONFERRONI = "p_val_Bonferroni_corrected"

relevant_statistical_results = [
    "volumetry_variable_tiv_level_Region",
    "volumetry_variable_tiv_level_Structure",
    "FA_variable_percent_above_thres_level_Region",
    "FA_variable_percent_above_thres_level_Structure",
    "MD_variable_p10_level_Region",
    "MD_variable_p10_level_Structure",
]

# %%
# load segmentation labels csv and ch2 original segmentations
labels_df = pd.read_csv(SEGMENTATION_LABELS_CSV, sep=";")

ch2_segmentation_nifti = load_nifti(CH2_SNYTHSEG_PARCELLATION)
ch2_voronoi_nifti = load_nifti(CH2_WM_VORONOI_PARCELLATION)

ch2_segmentation_arr = ch2_segmentation_nifti.get_fdata()
ch2_voronoi_arr = ch2_voronoi_nifti.get_fdata()

CH2_EXAMPLE_SEGMENTATION_DIR.mkdir(exist_ok=True, parents=True)
MAPPING_VISUALISATION_OUT_DIR.mkdir(exist_ok=True, parents=True)

# convert structure string-labels into integers
labels_df[STRUCTURE_ID] = pd.factorize(labels_df[STRUCTURE])[0] + 1
# NOTE: Missing values are set to -1; with +1, the missing background becomes 0 and hence maps
# identical
label_mapping_region2structure = dict(
    zip(labels_df[REGION_ID], labels_df[STRUCTURE_ID], strict=True)
)

# create mapping of label names with underscores to region IDs
labels_df[LABEL_UNDERSCORE] = labels_df[LABEL].str.replace("-", "_")
labels_df[LABEL_UNDERSCORE] = (
    labels_df[LABEL_UNDERSCORE].str.replace("3rd", "Third").str.replace("4th", "Fourth")
)

label_mapping_labelname2region_id = dict(
    zip(labels_df[LABEL_UNDERSCORE], labels_df[REGION_ID], strict=True)
)

# %%
# create ch2 images
###################
# create visualisable parcellations for region/structure segmention/voronoi labels of the
# ch2 example; all labels are remapped to make visualisation with standard scales feasible (e.g.
# from 0, 1, 1001, 2001 to 0,1,2,3)

ch2_segm_region_out = CH2_EXAMPLE_SEGMENTATION_DIR / "ch2_segm_region.nii.gz"
ch2_segm_structure_out = CH2_EXAMPLE_SEGMENTATION_DIR / "ch2_segm_structure.nii.gz"
ch2_segm_wm_out = CH2_EXAMPLE_SEGMENTATION_DIR / "ch2_segm_wm.nii.gz"
ch2_voronoi_region_out = CH2_EXAMPLE_SEGMENTATION_DIR / "ch2_voronoi_region.nii.gz"
ch2_voronoi_structure_out = (
    CH2_EXAMPLE_SEGMENTATION_DIR / "ch2_voronoi_structure.nii.gz"
)

output_files = [
    ch2_segm_region_out,
    ch2_segm_structure_out,
    ch2_segm_wm_out,
    ch2_voronoi_region_out,
    ch2_voronoi_structure_out,
]

if any(not file.exists() for file in output_files):
    ch2_segm_regions_remapped_arr = reassign_consecutive_labels(ch2_segmentation_arr)
    ch2_segm_regions_remapped_nifti = Nifti1Image(
        ch2_segm_regions_remapped_arr,
        affine=ch2_segmentation_nifti.affine,
        header=ch2_segmentation_nifti.header,
    )
    save_nifti(ch2_segm_regions_remapped_nifti, ch2_segm_region_out)

    ch2_voronoi_regions_remapped_arr = reassign_consecutive_labels(ch2_voronoi_arr)
    ch2_segm_regions_remapped_nifti = Nifti1Image(
        ch2_voronoi_regions_remapped_arr,
        affine=ch2_voronoi_nifti.affine,
        header=ch2_voronoi_nifti.header,
    )
    save_nifti(ch2_segm_regions_remapped_nifti, ch2_voronoi_region_out)

    ch2_segm_wm_arr = (ch2_voronoi_regions_remapped_arr != 0).astype(int)
    ch2_segm_wm_nifti = Nifti1Image(
        ch2_segm_wm_arr,
        affine=ch2_voronoi_nifti.affine,
        header=ch2_voronoi_nifti.header,
    )
    save_nifti(ch2_segm_wm_nifti, ch2_segm_wm_out)

    ch2_segm_structure_remapped_arr = np.vectorize(
        lambda x: label_mapping_region2structure.get(x, x)
    )(ch2_segmentation_arr)
    ch2_segm_structure_remapped_nifti = Nifti1Image(
        ch2_segm_structure_remapped_arr,
        affine=ch2_segmentation_nifti.affine,
        header=ch2_segmentation_nifti.header,
    )
    save_nifti(ch2_segm_structure_remapped_nifti, ch2_segm_structure_out)

    ch2_voronoi_structure_remapped_arr = np.vectorize(
        lambda x: label_mapping_region2structure.get(x, x)
    )(ch2_voronoi_arr)
    ch2_voronoi_structure_remapped_nifti = Nifti1Image(
        ch2_voronoi_structure_remapped_arr,
        affine=ch2_voronoi_nifti.affine,
        header=ch2_voronoi_nifti.header,
    )
    save_nifti(ch2_voronoi_structure_remapped_nifti, ch2_voronoi_structure_out)

else:
    print(
        f"Skipping: All ch2 example nifti already exist in {CH2_EXAMPLE_SEGMENTATION_DIR}"
    )


# %%
# create mixed models results niftis
####################################


def _map_labels(x, label_mapping):
    """Wrap function to address Pylance warnings."""
    return label_mapping.get(x, x)


for marker_result in relevant_statistical_results:
    marker_csv = MIXED_MODEL_OUTPUT_DIR / f"{marker_result}_mixed_models.csv"
    marker_df = pd.read_csv(marker_csv, sep=";")

    # map label names to region id as found in the segmentation
    marker_df[REGION_ID] = marker_df["marker"].map(label_mapping_labelname2region_id)

    # create new column with effect sizes that are siginficant (i.e. set non-significant ones to 0)
    marker_df[P_BONFERRONI] = marker_df["lrt_pvalue"] * len(marker_df)
    marker_df[R2_DELTA_SIGNIFICANT] = marker_df.apply(
        lambda row: row[R2_DELTA] if row[P_BONFERRONI] < ALHPA_LEVEL else 0, axis=1
    )

    ######
    # Volumetry - based on original synthseg segmentation
    if "volumetry" in marker_result:
        if "Region" in marker_result:
            label_mapping_region2effect = dict(
                zip(marker_df[REGION_ID], marker_df[R2_DELTA_SIGNIFICANT], strict=True)
            )

            mapping_arr = np.vectorize(_map_labels, excluded=["label_mapping"])(
                ch2_segmentation_arr, label_mapping_region2effect
            )
        elif "Structure" in marker_result:
            # clean column values; this is only necessary for this specific marker set
            marker_df["marker"] = marker_df["marker"].str.replace("_structure", "")

            structure_effect_size_map = dict(
                zip(marker_df["marker"], marker_df[R2_DELTA_SIGNIFICANT], strict=True)
            )
            labels_df_temp = labels_df.copy()
            labels_df_temp[R2_DELTA_SIGNIFICANT] = labels_df_temp[STRUCTURE].map(
                structure_effect_size_map
            )
            label_mapping_region2effect = dict(
                zip(
                    labels_df_temp[REGION_ID],
                    labels_df_temp[R2_DELTA_SIGNIFICANT],
                    strict=True,
                )
            )
            mapping_arr = np.vectorize(_map_labels, excluded=["label_mapping"])(
                ch2_segmentation_arr, label_mapping_region2effect
            )

        else:
            msg = f"No level assignable to {marker_result}"
            raise ValueError(msg)

        mapping_nifti = Nifti1Image(
            mapping_arr,
            affine=ch2_segmentation_nifti.affine,
            header=ch2_segmentation_nifti.header,
        )
    ######
    # FA/MD - based on WM voronoi segmentation
    elif any(s in marker_result for s in ["FA", "MD"]):
        if "Region" in marker_result:
            label_mapping_region2effect = dict(
                zip(marker_df[REGION_ID], marker_df[R2_DELTA_SIGNIFICANT], strict=True)
            )

            mapping_arr = np.vectorize(_map_labels, excluded=["label_mapping"])(
                ch2_voronoi_arr, label_mapping_region2effect
            )
        elif "Structure" in marker_result:
            structure_effect_size_map = dict(
                zip(marker_df["marker"], marker_df[R2_DELTA_SIGNIFICANT], strict=True)
            )
            labels_df_temp = labels_df.copy()
            labels_df_temp[R2_DELTA_SIGNIFICANT] = labels_df_temp[STRUCTURE].map(
                structure_effect_size_map
            )
            label_mapping_region2effect = dict(
                zip(
                    labels_df_temp[REGION_ID],
                    labels_df_temp[R2_DELTA_SIGNIFICANT],
                    strict=True,
                )
            )
            mapping_arr = np.vectorize(_map_labels, excluded=["label_mapping"])(
                ch2_voronoi_arr, label_mapping_region2effect
            )
        else:
            msg = f"No level assignable to {marker_result}"
            raise ValueError(msg)

        mapping_nifti = Nifti1Image(
            mapping_arr,
            affine=ch2_voronoi_nifti.affine,
            header=ch2_voronoi_nifti.header,
        )
    else:
        msg = f"No marker assignable to {marker_result}"
        raise ValueError(msg)

    marker_str_clean = marker_result.replace("_variable", "").replace("_level", "")
    outname = (
        MAPPING_VISUALISATION_OUT_DIR / f"mixed_model_effects_{marker_str_clean}.nii.gz"
    )
    save_nifti(mapping_nifti, outname=outname)

print("Done.")

# %%
