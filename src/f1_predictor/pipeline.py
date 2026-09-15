"""CLI entrypoint: run load -> feature -> train -> predict -> evaluate for one race.

Usage:
    python -m f1_predictor.pipeline --race monaco_gp
"""

from __future__ import annotations

import argparse
import logging

from sklearn.model_selection import train_test_split

from f1_predictor import evaluate
from f1_predictor.config import RaceConfig, load_race_config
from f1_predictor.data import loader
from f1_predictor.features.engineer import build_feature_table
from f1_predictor.models import baseline, predict, train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run(race: str) -> dict:
    config: RaceConfig = load_race_config(race)

    historical_laps = loader.load_historical_laps(config.historical)
    sector_times = loader.average_sector_times(historical_laps)
    rain_probability, temperature = loader.fetch_weather_forecast(config.weather)

    table = build_feature_table(config, sector_times, historical_laps, rain_probability, temperature)
    trainable = table.dropna(subset=["LapTime (s)"]).reset_index(drop=True)

    if trainable.empty:
        raise ValueError(
            f"No overlap between {config.name} qualifying drivers and {config.historical.year} "
            f"round {config.historical.round} historical data."
        )

    train_idx, test_idx = train_test_split(
        trainable.index, test_size=config.model.test_size, random_state=config.model.random_state
    )
    train_table = trainable.loc[train_idx]
    test_table = trainable.loc[test_idx]

    pipeline, columns = train.fit(train_table, config.model)

    model_metrics = evaluate.regression_metrics(test_table["LapTime (s)"], pipeline.predict(test_table[columns]))
    baseline_metrics = evaluate.regression_metrics(test_table["LapTime (s)"], test_table["QualifyingTime (s)"])
    lift = evaluate.ranking_lift(model_metrics, baseline_metrics)

    results = predict.predict_race_times(pipeline, table, columns)
    podium = predict.podium(results)

    print(f"\nPredicted {config.name} result\n")
    print(results[["Driver", "PredictedRaceTime (s)"]].to_string(index=False))
    print(
        f"\nModel   -> MAE: {model_metrics['mae']:.2f}s | RMSE: {model_metrics['rmse']:.2f}s | "
        f"Spearman: {model_metrics['spearman']:.2f}"
    )
    print(
        f"Baseline-> MAE: {baseline_metrics['mae']:.2f}s | RMSE: {baseline_metrics['rmse']:.2f}s | "
        f"Spearman: {baseline_metrics['spearman']:.2f}  (predicting quali order as-is)"
    )
    print(f"Lift over baseline: {lift['spearman_lift']:+.2f} spearman, {lift['mae_improvement_pct']:+.1f}% MAE")

    print("\nPredicted podium:")
    for position, (_, row) in zip(["P1", "P2", "P3"], podium.iterrows()):
        print(f"  {position}: {row['Driver']} ({row['PredictedRaceTime (s)']:.2f}s)")

    return {
        "config": config,
        "results": results,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "lift": lift,
        "pipeline": pipeline,
        "columns": columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the F1 race prediction pipeline for one race.")
    parser.add_argument("--race", required=True, help="Race config slug, e.g. 'monaco_gp' (see configs/races/)")
    args = parser.parse_args()
    run(args.race)


if __name__ == "__main__":
    main()
