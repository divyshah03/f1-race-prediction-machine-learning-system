"""Train one model across every race/circuit instead of nine separate per-race models.

Adds `circuit` as a categorical feature, benchmarks GradientBoosting / XGBoost /
LightGBM / CatBoost with a time-aware (walk-forward, by historical round) split
instead of a random split, and compares every candidate against the naive
qualifying-order baseline. See Phase 2 of todo.md.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from f1_predictor import evaluate, tracking
from f1_predictor.config import ModelConfig, available_races, load_race_config
from f1_predictor.data import loader
from f1_predictor.explain import build_shap_summary_figure
from f1_predictor.features.engineer import build_feature_table
from f1_predictor.models import train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_MODEL_TYPES = ["gradient_boosting", "xgboost", "lightgbm", "catboost"]


def build_unified_table(races: list[str] | None = None) -> pd.DataFrame:
    """Load every race, engineer features, and concatenate into one long table."""
    races = races or available_races()
    frames = []

    for race in races:
        config = load_race_config(race)
        historical_laps = loader.load_historical_laps(config.historical)
        sector_times = loader.average_sector_times(historical_laps)
        rain_probability, temperature = loader.fetch_weather_forecast(config.weather)

        table = build_feature_table(config, sector_times, historical_laps, rain_probability, temperature)
        table["round"] = config.historical.round
        table["race"] = race
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["LapTime (s)"]).reset_index(drop=True)


def walk_forward_split(table: pd.DataFrame, held_out_rounds: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest rounds of the season, test on the latest ones.

    This is the time-aware alternative to a random train/test split: it never lets
    the model see a "future" race's outcome while training, which is what a random
    split silently does across a season.
    """
    rounds = sorted(table["round"].unique())
    if len(rounds) <= held_out_rounds:
        raise ValueError("Not enough distinct rounds to hold any out for a walk-forward split.")

    split_rounds = set(rounds[-held_out_rounds:])
    test_mask = table["round"].isin(split_rounds)
    return table.loc[~test_mask].reset_index(drop=True), table.loc[test_mask].reset_index(drop=True)


def benchmark_models(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    model_types: list[str] = CANDIDATE_MODEL_TYPES,
    track_with_mlflow: bool = True,
) -> pd.DataFrame:
    """Fit each candidate model type on the same walk-forward split and compare metrics."""
    rows = []
    baseline_metrics = evaluate.regression_metrics(
        test_table["LapTime (s)"], test_table["QualifyingTimeRaw (s)"]
    )
    rows.append({"model": "baseline_quali_order", **baseline_metrics})

    for model_type in model_types:
        model_config = ModelConfig(type=model_type, n_estimators=200, learning_rate=0.1, random_state=39)
        try:
            pipeline, columns = train.fit(train_table, model_config, include_circuit=True)
        except ImportError:
            logger.warning("Skipping %s: package not installed.", model_type)
            continue

        predictions = pipeline.predict(test_table[columns])
        metrics = evaluate.regression_metrics(test_table["LapTime (s)"], predictions)
        rows.append({"model": model_type, **metrics})

        if track_with_mlflow:
            try:
                shap_figure = build_shap_summary_figure(pipeline, train_table, columns)
            except Exception:
                logger.warning(
                    "SHAP summary plot failed for %s; logging the run without it.",
                    model_type,
                    exc_info=True,
                )
                shap_figure = None

            try:
                tracking.log_model_run(
                    run_name=model_type,
                    params=dataclasses_to_dict(model_config),
                    metrics=metrics,
                    model=pipeline,
                    shap_figure=shap_figure,
                )
            except Exception:
                logger.warning(
                    "MLflow logging failed for %s; continuing without it.", model_type, exc_info=True
                )

    return pd.DataFrame(rows)


def dataclasses_to_dict(model_config: ModelConfig) -> dict:
    return {
        "model_type": model_config.type,
        "n_estimators": model_config.n_estimators,
        "learning_rate": model_config.learning_rate,
        "max_depth": model_config.max_depth,
        "random_state": model_config.random_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark a unified cross-race model.")
    parser.add_argument("--held-out-rounds", type=int, default=2, help="Number of latest rounds to test on.")
    args = parser.parse_args()

    table = build_unified_table()
    train_table, test_table = walk_forward_split(table, held_out_rounds=args.held_out_rounds)

    logger.info(
        "Unified table: %d rows across %d races -> %d train / %d test (walk-forward on round).",
        len(table),
        table["race"].nunique(),
        len(train_table),
        len(test_table),
    )

    leaderboard = benchmark_models(train_table, test_table).sort_values("mae")
    print("\nModel benchmark (walk-forward split, sorted by MAE):\n")
    print(leaderboard.to_string(index=False))

    best = leaderboard.iloc[0]
    baseline_row = leaderboard[leaderboard["model"] == "baseline_quali_order"].iloc[0]
    lift = evaluate.ranking_lift(best.to_dict(), baseline_row.to_dict())
    print(
        f"\nBest model ({best['model']}) lift over baseline: "
        f"{lift['spearman_lift']:+.2f} spearman, {lift['mae_improvement_pct']:+.1f}% MAE"
    )


if __name__ == "__main__":
    main()
