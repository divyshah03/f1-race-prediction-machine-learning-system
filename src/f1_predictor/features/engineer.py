"""Feature engineering: turn qualifying + historical + weather data into a model-ready table.

This replaces the bespoke ``prepare_training_data`` / reference-data functions that
used to be duplicated (with small inconsistencies) across all nine race scripts.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from f1_predictor.config import RaceConfig

SECTOR_COLUMNS = ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "TotalSectorTime (s)"]
IDENTIFIER_COLUMNS = {"Driver", "Team", "circuit", "LapTime (s)", "QualifyingTimeRaw (s)"}


def _team_performance_score(config: RaceConfig) -> dict[str, float]:
    if not config.team_points:
        return {}
    max_points = max(config.team_points.values()) or 1.0
    return {team: points / max_points for team, points in config.team_points.items()}


def build_feature_table(
    config: RaceConfig,
    sector_times: pd.DataFrame,
    historical_laps: pd.DataFrame,
    rain_probability: float,
    temperature: float,
) -> pd.DataFrame:
    """Merge qualifying, sector, weather and reference data into one per-driver table.

    Every optional feature (wet performance, team strength, clean-air pace, ...) is
    only added when the race config actually provides it, so a unified model trained
    across races naturally sees NaN for races that don't have a given signal.
    """
    table = pd.DataFrame(
        {
            "Driver": [d.code for d in config.drivers],
            "QualifyingTime (s)": [d.qualifying_time for d in config.drivers],
        }
    )
    # Keep the untransformed qualifying time around for baseline comparisons, since
    # "QualifyingTime (s)" below may get weather-adjusted and/or squared for the model.
    table["QualifyingTimeRaw (s)"] = table["QualifyingTime (s)"]

    if sector_times is not None and not sector_times.empty:
        table = table.merge(sector_times[["Driver", *SECTOR_COLUMNS]], on="Driver", how="left")

    table["RainProbability"] = rain_probability
    table["Temperature"] = temperature

    if config.wet_performance_factor:
        table["WetPerformanceFactor"] = table["Driver"].map(config.wet_performance_factor)
        if rain_probability >= config.weather.rain_threshold:
            table["QualifyingTime (s)"] = table["QualifyingTime (s)"] * table["WetPerformanceFactor"].fillna(
                1.0
            )

    if config.qualifying_time_transform == "square":
        table["QualifyingTime (s)"] = table["QualifyingTime (s)"] ** 2

    if config.driver_team:
        table["Team"] = table["Driver"].map(config.driver_team)
        team_score = _team_performance_score(config)
        table["TeamPerformanceScore"] = table["Team"].map(team_score)

    if config.clean_air_race_pace:
        table["CleanAirRacePace (s)"] = table["Driver"].map(config.clean_air_race_pace)

    if config.average_position_change:
        table["AveragePositionChange"] = table["Driver"].map(config.average_position_change)

    if config.season_points:
        table["SeasonPoints"] = table["Driver"].map(config.season_points)

    if config.average_2025_performance:
        table["Average2025Performance"] = table["Driver"].map(config.average_2025_performance)

    if config.last_year_winner:
        table["LastYearWinner"] = (table["Driver"] == config.last_year_winner).astype(int)

    table["circuit"] = config.circuit

    if historical_laps is not None and not historical_laps.empty:
        target = historical_laps.groupby("Driver")["LapTime (s)"].mean()
        table["LapTime (s)"] = table["Driver"].map(target)

    return table


def numeric_feature_columns(table: pd.DataFrame) -> list[str]:
    """Numeric feature columns available for modeling (excludes identifiers/target)."""
    return [
        col
        for col in table.columns
        if col not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(table[col])
    ]


def build_preprocessing_pipeline(
    numeric_features: list[str], include_circuit: bool = False
) -> ColumnTransformer:
    """A ColumnTransformer that median-imputes numeric features and one-hot encodes circuit."""
    transformers = [("numeric", SimpleImputer(strategy="median"), numeric_features)]
    if include_circuit:
        transformers.append(("circuit", OneHotEncoder(handle_unknown="ignore"), ["circuit"]))
    return ColumnTransformer(transformers)
