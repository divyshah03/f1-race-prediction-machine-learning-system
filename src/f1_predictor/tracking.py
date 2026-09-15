"""MLflow experiment tracking helpers.

Wraps a single model run with `mlflow.start_run()` and logs hyperparameters,
regression metrics, and the fitted model artifact. Uses a local `./mlruns`
store unless `MLFLOW_TRACKING_URI` is set (see `.env.example`).
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


def log_model_run(
    run_name: str, params: dict, metrics: dict, model=None, artifact_path: str = "model"
) -> None:
    """Log one training run's hyperparameters, metrics, and (optionally) the fitted model."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, artifact_path)
            except Exception:
                logger.warning("Could not log model artifact for run %r.", run_name, exc_info=True)
