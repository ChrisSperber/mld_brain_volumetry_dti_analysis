"""Utils for elastic net regression."""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

from brain_stats_tools.config import RNG_SEED

N_PREDICTION_REPS = 250  # number of repeated cross-validations per marker set
TEST_SIZE_RATIO = 0.2
NONZERO_COEFF_THRESHOLD = 1e-4  # threshold to identify non-zero coefficients

BAYESIAN_OPTIMISATION_ITERATIONS = 50


def train_test_split_indices(
    n: int, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate indices for train and test folds.

    Args:
        n (int): Total sample size
        test_ratio (float): Ratio of sample assigned to test fold.
        seed (int): RNG seed.

    Returns:
        Tuple: Arrays of training and test indices.

    """
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    return train_idx, test_idx


def fit_elastic_net_bayes_opt(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = BAYESIAN_OPTIMISATION_ITERATIONS,
) -> ElasticNet:
    """Fit ElasticNet using Bayesian optimization to tune hyperparameters.

    Features are standardised via make_pipeline to obtain comparable results with regularisation.

    Args:
        x (np.ndarray): Predictors
        y (np.ndarray): Target Variable
        n_iter (int, optional): Number of iterations for Bayesian optimisation.
            Defaults to BAYESIAN_OPTIMISATION_ITERATIONS.

    Returns:
        ElasticNet: Optimised elastic net model.

    """
    pipe = make_pipeline(StandardScaler(), ElasticNet(max_iter=5000))

    param_space = {
        "elasticnet__alpha": Real(1e-3, 1e2, prior="log-uniform"),
        "elasticnet__l1_ratio": Real(0.1, 0.9),
    }

    opt: BayesSearchCV = BayesSearchCV(
        estimator=pipe,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=4,
        scoring="r2",
        n_jobs=-1,
        random_state=RNG_SEED,
        verbose=0,
    )
    opt.fit(x, y)
    return opt.best_estimator_  # pyright: ignore[reportAttributeAccessIssue]
