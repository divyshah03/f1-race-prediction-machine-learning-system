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


def grouped_regression_metrics(
    df: pd.DataFrame, group_col: str, true_col: str, pred_col: str
) -> dict[str, float]:
    """MAE/RMSE pooled across `df`, but Spearman averaged *within* each `group_col` group.

    Finishing order only ever matters within a single race: a driver's predicted lap
    time is never compared against a driver's from a different circuit. Pooling rows
    from several races (e.g. a walk-forward test set spanning Monaco and Abu Dhabi)
    before computing one Spearman correlation conflates "ranks drivers correctly
    within a race" with "happens to separate two circuits' overall pace," which is a
    different, unrelated question and can flip the sign of the metric entirely for a
    model whose absolute predictions cluster similarly across circuits even though its
    within-race ranking is sound. MAE/RMSE don't have this problem -- an absolute
    time error is comparable across races -- so those stay pooled.
    """
    pooled = regression_metrics(df[true_col], df[pred_col])

    correlations = []
    for _, group in df.groupby(group_col):
        group_metrics = regression_metrics(group[true_col], group[pred_col])
        if len(group) >= 2 and np.isfinite(group_metrics["spearman"]):
            correlations.append(group_metrics["spearman"])

    spearman = float(np.mean(correlations)) if correlations else float("nan")
    return {"mae": pooled["mae"], "rmse": pooled["rmse"], "spearman": spearman}


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
