"""Fetch and analyse FA structure data of repeated sessions.

This script performs a sensitivity analysis to shed further light on the brain mapping results.
Specifically for the FA %voxels above threshold measure on structure level (i.e. looking at larger
structures like lobes), intraindividual changes are evaluated. For patient that were repeatedly
tested at differing time points, the Kendall's correlation between the change in GMFC and
MRIscore/FA brain marker is computed.

Note: The sample only included 15 intersession measures, hence results should be interpreted
carefully.

Outputs:
    - CSV with Kendall's Tau and p-value per marker
"""

# %%
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

from brain_stats_tools.config import PREPARED_DATA_DIR
from brain_stats_tools.utils import Cols, LongDFCols

CLINICAL_DATA_CLEANED_CSV = Path(__file__).parent / "b_fetch_clinical_data.csv"

FA_BRAIN_MARKER_CSV = "metrics_FA_variable_percent_above_thres_level_Structure.csv"

RELEVANT_STRUCTURES = [
    "Frontal",
    "Temporal",
    "Parietal",
    "Occipital",
    "Insula",
    "Subcortical_Deep_Gray",
    "Cingulate",
]

DELTA_GMFC = "Delta_GMFC"
DELTA_MRISCORE = "Delta_MRIScore"

# %%
# load clinical data csv and find repeated sessions
full_data_df = pd.read_csv(CLINICAL_DATA_CLEANED_CSV, sep=";")

# keep only patients with multiple sessions
data_df = full_data_df[
    full_data_df.groupby(Cols.SUBJECT_ID)[Cols.SUBJECT_ID].transform("size") > 1
]

data_df[LongDFCols.BASENAME] = (
    "subject_"
    + data_df[Cols.SUBJECT_ID].astype(str)
    + "_date_"
    + data_df[Cols.DATE_TAG].astype(str)
)

# %%
# load brain markers
fa_structure_df = pd.read_csv(PREPARED_DATA_DIR / FA_BRAIN_MARKER_CSV, sep=";")
# drop controls
fa_structure_df = fa_structure_df[
    ~fa_structure_df[LongDFCols.BASENAME].str.contains("MLD")
]

# merge dfs and sort by subject and Date
# --> consecutive sessions are shown on consecutive rows
data_df = data_df.merge(
    fa_structure_df,
    how="inner",
    on=[LongDFCols.BASENAME],
    validate="one_to_one",
).sort_values([Cols.SUBJECT_ID, Cols.DATE_TAG], ascending=[True, True])
data_df = data_df.reset_index(drop=True)

# %%
# fetch data from repeated measurements
delta_list = []
rows = list(data_df.itertuples(index=True))

for current, nxt in zip_longest(rows, rows[1:], fillvalue=None):
    if nxt is None:
        # handle last row
        continue
    subj_current = getattr(current, Cols.SUBJECT_ID)
    subj_next = getattr(nxt, Cols.SUBJECT_ID)
    if subj_current != subj_next:
        # skip if not a repeated session by same subject
        continue
    session_delta = {
        Cols.SUBJECT_ID: subj_current,
        "Date_1": getattr(current, Cols.DATE_TAG),
        "Date_2": getattr(nxt, Cols.DATE_TAG),
        DELTA_GMFC: (getattr(nxt, Cols.GMFC) - getattr(current, Cols.GMFC)),
        DELTA_MRISCORE: (
            getattr(nxt, Cols.MRI_SCORE) - getattr(current, Cols.MRI_SCORE)
        ),
    }
    for structure in RELEVANT_STRUCTURES:
        structure_delta_str = f"Delta_{structure}"
        session_delta[structure_delta_str] = getattr(nxt, structure) - getattr(
            current, structure
        )
    delta_list.append(session_delta)

delta_df = pd.DataFrame(delta_list)

# %%
# compute correlations
relevant_structure_deltas = ["Delta_" + s for s in RELEVANT_STRUCTURES]

correl_var_cols = [DELTA_MRISCORE, *relevant_structure_deltas]
correl_results = []

for var_col in correl_var_cols:
    var_series = delta_df[var_col]
    gmfc_series = delta_df[DELTA_GMFC]
    tau, p = kendalltau(var_series, gmfc_series)
    correl_result = {
        "Variable": var_col,
        "Tau": tau,
        "p-value": p,
    }
    correl_results.append(correl_result)

correl_results_df = pd.DataFrame(correl_results)

# %%
output_name = Path(__file__).with_suffix(".csv")
correl_results_df.to_csv(output_name, index=False, sep=";")

# %%
