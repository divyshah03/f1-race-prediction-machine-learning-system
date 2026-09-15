"""Prediction helpers: score a fitted pipeline against a feature table."""

from __future__ import annotations

import numpy as np
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


def podium_probabilities(
    result: pd.DataFrame,
    residual_std: float,
    n_simulations: int = 5000,
    random_state: int = 39,
) -> pd.DataFrame:
    """Monte Carlo win/podium probabilities, not just a single point prediction.

    A point-predicted race time hides how confident the model actually is: two
    drivers 0.05s apart in predicted time are effectively a coin flip, not a
    guaranteed finishing order. This perturbs each driver's predicted time by
    the model's own held-out residual noise (assumed ~ N(0, residual_std)) many
    times and reports how often each driver lands on the podium / takes P1.
    """
    rng = np.random.default_rng(random_state)
    predicted = result["PredictedRaceTime (s)"].to_numpy()
    drivers = result["Driver"].to_numpy()
    n_drivers = len(predicted)

    noise_scale = max(float(residual_std), 0.05)
    simulated_times = predicted[None, :] + rng.normal(0.0, noise_scale, size=(n_simulations, n_drivers))
    finishing_order = np.argsort(simulated_times, axis=1)

    podium_counts = np.bincount(finishing_order[:, : min(3, n_drivers)].ravel(), minlength=n_drivers)
    win_counts = np.bincount(finishing_order[:, 0], minlength=n_drivers)

    return (
        pd.DataFrame(
            {
                "Driver": drivers,
                "PodiumProbability": podium_counts / n_simulations,
                "WinProbability": win_counts / n_simulations,
            }
        )
        .sort_values("PodiumProbability", ascending=False)
        .reset_index(drop=True)
    )
