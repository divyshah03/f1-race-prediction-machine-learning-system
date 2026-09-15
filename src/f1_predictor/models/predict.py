"""Prediction helpers: score a fitted pipeline against a feature table."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_race_times(pipeline: Pipeline, table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Predict race time for every driver in ``table`` and sort fastest first."""
    predictions = pipeline.predict(table[columns])
    result = table.copy()
    result["PredictedRaceTime (s)"] = predictions
    return result.sort_values("PredictedRaceTime (s)").reset_index(drop=True)


def podium(result: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Top-n predicted finishers."""
    return result.head(n)[["Driver", "PredictedRaceTime (s)"]]
