import pandas as pd
import pytest

from f1_predictor.config import DriverEntry, HistoricalSessionConfig, ModelConfig, RaceConfig, WeatherConfig


@pytest.fixture
def sample_historical_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": ["VER", "VER", "NOR", "NOR", "LEC", "LEC"],
            "LapTime (s)": [90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
            "Sector1Time (s)": [30.0, 30.5, 31.0, 31.5, 32.0, 32.5],
            "Sector2Time (s)": [30.0, 30.0, 30.5, 30.5, 31.0, 31.0],
            "Sector3Time (s)": [30.0, 30.5, 30.5, 31.0, 31.0, 31.5],
        }
    )


@pytest.fixture
def sample_race_config() -> RaceConfig:
    return RaceConfig(
        name="Fixture Grand Prix",
        circuit="fixture_circuit",
        season=2025,
        historical=HistoricalSessionConfig(year=2024, round=1, session="R"),
        weather=WeatherConfig(
            latitude=0.0, longitude=0.0, forecast_time="2025-01-01 00:00:00", rain_threshold=0.75
        ),
        model=ModelConfig(
            type="gradient_boosting", n_estimators=10, learning_rate=0.5, test_size=0.34, random_state=1
        ),
        drivers=[
            DriverEntry(code="VER", qualifying_time=80.0),
            DriverEntry(code="NOR", qualifying_time=81.0),
            DriverEntry(code="LEC", qualifying_time=82.0),
        ],
    )
