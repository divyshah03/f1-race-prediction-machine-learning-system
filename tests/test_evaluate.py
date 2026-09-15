import numpy as np
import pandas as pd
import pytest

from f1_predictor import evaluate


def test_regression_metrics_perfect_prediction_has_zero_error():
    y_true = pd.Series([90.0, 91.0, 92.0])
    y_pred = np.array([90.0, 91.0, 92.0])

    metrics = evaluate.regression_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["spearman"] == 1.0


def test_regression_metrics_inverted_order_has_negative_correlation():
    y_true = pd.Series([90.0, 91.0, 92.0])
    y_pred = np.array([92.0, 91.0, 90.0])

    metrics = evaluate.regression_metrics(y_true, y_pred)

    assert metrics["spearman"] == -1.0


def test_regression_metrics_drops_nan_pairs_instead_of_crashing():
    y_true = pd.Series([90.0, 91.0, 92.0])
    y_pred = np.array([90.0, np.nan, 92.0])

    metrics = evaluate.regression_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0


def test_ranking_lift_reports_positive_lift_when_model_beats_baseline():
    model_metrics = {"mae": 1.0, "rmse": 1.5, "spearman": 0.9}
    baseline_metrics = {"mae": 2.0, "rmse": 2.5, "spearman": 0.5}

    lift = evaluate.ranking_lift(model_metrics, baseline_metrics)

    assert lift["spearman_lift"] == pytest.approx(0.4)
    assert lift["mae_improvement_pct"] == pytest.approx(50.0)
