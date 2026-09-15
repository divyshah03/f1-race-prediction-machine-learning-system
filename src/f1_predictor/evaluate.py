"""Evaluation metrics: MAE, RMSE, and rank-correlation (Spearman) between predicted
and actual finishing order, plus the lift of a model over the naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / Spearman between true and predicted values.

    Drops any pair where either side is NaN first -- e.g. a driver with no
    recorded qualifying time (like Monaco's wet-weather no-time) still has
    historical lap data, so it can land in a baseline comparison with a NaN
    "prediction" that would otherwise crash sklearn's metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr, y_pred_arr = y_true_arr[mask], y_pred_arr[mask]

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = mean_squared_error(y_true_arr, y_pred_arr) ** 0.5
    correlation, _ = spearmanr(y_true_arr, y_pred_arr)
    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(correlation)}


def ranking_lift(model_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> dict[str, float]:
    """How much better the model's rank-correlation/MAE are than the naive baseline's."""
    mae_improvement_pct = (
        (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100
        if baseline_metrics["mae"]
        else float("nan")
    )
    return {
        "spearman_lift": model_metrics["spearman"] - baseline_metrics["spearman"],
        "mae_improvement_pct": mae_improvement_pct,
    }
