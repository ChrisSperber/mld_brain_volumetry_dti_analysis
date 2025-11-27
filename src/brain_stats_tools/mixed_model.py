"""Mixed effects regression model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from brain_stats_tools.config import RNG_SEED
from brain_stats_tools.utils import Cols

N_BOOTSTRAPS_DELTA_R2 = 500


@dataclass
class MixedMarkerResult:
    """Structured output of mixed model analysis."""

    r2_null: float
    r2_full: float
    r2_delta: float
    lrt_stat: float
    lrt_df: int
    lrt_pvalue: float
    beta_marker: float
    beta_marker_se: float
    beta_marker_pvalue: float
    r2_delta_ci_low: float | None = None
    r2_delta_ci_high: float | None = None
    n_bootstrap: int | None = None


def _marginal_r2(result) -> float:
    """Compute Nakagawa-style marginal R² for a random-intercept mixed model.

    R2_marg = Var(fixed part) / (Var(fixed part) + Var(random) + Var(residual))

    Assumes:
        - random intercept only
    """
    # fixed-effects design matrix and parameters
    exog_fe = result.model.exog  # n x p
    beta_fe = result.fe_params  # length p

    # fixed-effects linear predictor
    eta_fe = np.dot(exog_fe, beta_fe)  # shape (n,)

    var_fe = np.var(eta_fe, ddof=1)

    # variance of random intercept(s)
    # For a single random intercept, cov_re is 1x1
    var_re = float(np.trace(result.cov_re.values))

    # residual variance
    var_resid = float(result.scale)

    denom = var_fe + var_re + var_resid
    if denom == 0:
        return np.nan

    return float(var_fe / denom)


def fit_marker_mixed_model(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    marker_colname: str,
    target_var_colname: str,
    subject_colname: str = Cols.SUBJECT_ID,
    age_colname: str = Cols.AGE,
    sex_colname: str = Cols.SEX,
    n_bootstrap: int = 0,
    ci_level: float = 0.95,
    random_state: int = RNG_SEED,
) -> MixedMarkerResult:
    """Fit two linear mixed models with random intercept for subject.

      Null:  target ~ age + sex + (1 | subject)
      Full:  target ~ age + sex + marker + (1 | subject)

    and return marginal R²s and likelihood ratio test for the marker.

    Parameters
    ----------
    df : DataFrame
        One row per measurement (i.e. repeated measures on different rows)
    marker_colname : str
        Column with the imaging marker.
    target_var_colname : str
        Column with the outcome variable.
    subject_colname : str
        Column with subject ID (random-intercept grouping factor).
    age_colname : str
        Column with age (continuous).
    sex_colname : str
        Column with sex (categorical).
    n_bootstrap: int
        Number of bootstraps to estimate ΔR² CI
    ci_level: float
        Confidence interval, defaults to 0.95
    random_state: int
        RNG seed, defaults to config RNG_SEED

    Returns
    -------
    MixedMarkerResult
        Contains marginal R² (null, full, delta), LRT statistic, df, p,
        and the marker coefficient with its Wald SE and p-value.

    """
    # Drop rows with missing values in any relevant column
    needed_cols = [
        subject_colname,
        target_var_colname,
        age_colname,
        sex_colname,
        marker_colname,
    ]
    df_model = df.dropna(subset=needed_cols).copy()

    formula_null = f"{target_var_colname} ~ {age_colname} + {sex_colname}"
    formula_full = (
        f"{target_var_colname} ~ {age_colname} + {sex_colname} + {marker_colname}"
    )

    # Fit models with ML (reml=False) so that LRT is valid
    model_null = smf.mixedlm(
        formula_null,
        data=df_model,
        groups=df_model[subject_colname],
    )
    result_null = model_null.fit(reml=False)

    model_full = smf.mixedlm(
        formula_full,
        data=df_model,
        groups=df_model[subject_colname],
    )
    result_full = model_full.fit(reml=False)

    # Likelihood ratio test: full vs null
    lr_stat = 2.0 * (result_full.llf - result_null.llf)
    df_diff = 1  # models differ by exactly one parameter: the marker
    p_lrt = float(chi2.sf(lr_stat, df=df_diff))

    # Marginal R² for each model
    r2_null = _marginal_r2(result_null)
    r2_full = _marginal_r2(result_full)
    r2_delta = r2_full - r2_null

    # Marker coefficient and Wald p-value from full model
    beta_marker = float(result_full.params[marker_colname])
    beta_marker_se = float(result_full.bse[marker_colname])
    beta_marker_pvalue = float(result_full.pvalues[marker_colname])

    # ---- Bootstrap CI for ΔR² (cluster bootstrap over subjects) ----
    r2_delta_ci_low: float | None = None
    r2_delta_ci_high: float | None = None

    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(random_state)

        subjects = df_model[subject_colname].unique()
        n_subjects = subjects.shape[0]
        r2_delta_boot = []

        for _ in range(n_bootstrap):
            # resample subjects with replacement
            boot_subjects = rng.choice(subjects, size=n_subjects, replace=True)

            # build bootstrap sample by concatenating rows of selected subjects
            boot_df = pd.concat(
                [df_model[df_model[subject_colname] == s] for s in boot_subjects],
                ignore_index=True,
            )

            try:
                boot_null = smf.mixedlm(
                    formula_null,
                    data=boot_df,
                    groups=boot_df[subject_colname],
                ).fit(reml=False)

                boot_full = smf.mixedlm(
                    formula_full,
                    data=boot_df,
                    groups=boot_df[subject_colname],
                ).fit(reml=False)

                boot_r2_null = _marginal_r2(boot_null)
                boot_r2_full = _marginal_r2(boot_full)
                r2_delta_boot.append(boot_r2_full - boot_r2_null)
            except Exception:  # noqa: S112
                # If a bootstrap sample fails to converge, skip it
                continue

        if r2_delta_boot:
            alpha = 1.0 - ci_level
            lower_q = alpha / 2.0
            upper_q = 1.0 - alpha / 2.0
            r2_delta_ci_low = float(np.quantile(r2_delta_boot, lower_q))
            r2_delta_ci_high = float(np.quantile(r2_delta_boot, upper_q))

    return MixedMarkerResult(
        r2_null=r2_null,
        r2_full=r2_full,
        r2_delta=r2_delta,
        lrt_stat=lr_stat,
        lrt_df=df_diff,
        lrt_pvalue=p_lrt,
        beta_marker=beta_marker,
        beta_marker_se=beta_marker_se,
        beta_marker_pvalue=beta_marker_pvalue,
        r2_delta_ci_low=r2_delta_ci_low,
        r2_delta_ci_high=r2_delta_ci_high,
        n_bootstrap=n_bootstrap,
    )
