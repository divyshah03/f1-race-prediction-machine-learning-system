"""MLflow experiment tracking helpers.

Wraps a single model run with `mlflow.start_run()` and logs hyperparameters,
regression metrics, and the fitted model artifact.

Defaults MLFLOW_TRACKING_URI to a *relative* local sqlite path
("sqlite:///mlflow.db") if not already set. MLflow 3.x's default behavior is to
resolve an unset tracking URI into an absolute path derived from the current
working directory; if that path contains characters like an apostrophe or a
space, the resulting sqlite URI gets percent-encoded in a way MLflow's own
sqlalchemy connection fails to parse, and the run silently never persists to
disk (metrics still print correctly, they just don't land in a queryable
store). A relative URI sidesteps that resolution entirely. Override with your
own MLFLOW_TRACKING_URI in `.env` if you want a different backend.
"""

from __future__ import annotations

import logging
import os

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

if not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

# mlflow's default sklearn serializer (skops) refuses to save any type it doesn't
# already know about, as a security measure against loading arbitrary pickled code.
# Our pipelines only ever contain these library classes, so trusting them here is
# safe -- it's the mechanism skops_trusted_types exists for.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "collections.OrderedDict",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "catboost.core.CatBoostRegressor",
]


def log_model_run(
    run_name: str, params: dict, metrics: dict, model=None, artifact_path: str = "model"
) -> None:
    """Log one training run's hyperparameters, metrics, and (optionally) the fitted model."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, artifact_path, skops_trusted_types=SKOPS_TRUSTED_TYPES)
            except Exception:
                logger.warning("Could not log model artifact for run %r.", run_name, exc_info=True)
