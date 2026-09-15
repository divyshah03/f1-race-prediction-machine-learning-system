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


def test_grouped_regression_metrics_averages_spearman_within_each_group():
    # Two "races" with disjoint, non-overlapping value ranges. Within each race the
    # prediction ranks drivers perfectly (spearman == 1.0), but pooling both races
    # before computing one correlation would score this as strongly *negative*,
    # since sorting by predicted value no longer matches sorting by true value once
    # the two races' ranges are interleaved.
    df = pd.DataFrame(
        {
            "race": ["a", "a", "a", "b", "b", "b"],
            "true": [10.0, 11.0, 12.0, 110.0, 111.0, 112.0],
            "pred": [50.0, 51.0, 52.0, 1.0, 2.0, 3.0],
        }
    )

    pooled = evaluate.regression_metrics(df["true"], df["pred"])
    grouped = evaluate.grouped_regression_metrics(df, "race", "true", "pred")

    assert pooled["spearman"] < 0
    assert grouped["spearman"] == pytest.approx(1.0)
    assert grouped["mae"] == pytest.approx(pooled["mae"])
    assert grouped["rmse"] == pytest.approx(pooled["rmse"])


def test_grouped_regression_metrics_skips_single_row_groups():
    df = pd.DataFrame(
        {
            "race": ["a", "a", "b"],
            "true": [10.0, 11.0, 50.0],
            "pred": [10.0, 11.0, 50.0],
        }
    )

    grouped = evaluate.grouped_regression_metrics(df, "race", "true", "pred")

    assert grouped["spearman"] == pytest.approx(1.0)


def test_ranking_lift_reports_positive_lift_when_model_beats_baseline():
    model_metrics = {"mae": 1.0, "rmse": 1.5, "spearman": 0.9}
    baseline_metrics = {"mae": 2.0, "rmse": 2.5, "spearman": 0.5}

    lift = evaluate.ranking_lift(model_metrics, baseline_metrics)

    assert lift["spearman_lift"] == pytest.approx(0.4)
    assert lift["mae_improvement_pct"] == pytest.approx(50.0)
