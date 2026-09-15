from unittest.mock import MagicMock, patch

import pandas as pd

from f1_predictor.config import HistoricalSessionConfig, WeatherConfig
from f1_predictor.data import loader


def test_load_historical_laps_converts_timedeltas_to_seconds(monkeypatch):
    raw_laps = pd.DataFrame(
        {
            "Driver": ["VER", "NOR"],
            "LapTime": pd.to_timedelta([90.5, 91.2], unit="s"),
            "Sector1Time": pd.to_timedelta([30.0, 30.5], unit="s"),
            "Sector2Time": pd.to_timedelta([30.0, 30.0], unit="s"),
            "Sector3Time": pd.to_timedelta([30.5, 30.7], unit="s"),
        }
    )

    fake_session = MagicMock()
    fake_session.laps = raw_laps
    fake_session.load = MagicMock()

    monkeypatch.setattr(loader, "_cache_initialized", True)

    with patch("f1_predictor.data.loader.fastf1.get_session", return_value=fake_session) as get_session:
        result = loader.load_historical_laps(HistoricalSessionConfig(year=2024, round=1, session="R"))

    get_session.assert_called_once_with(2024, 1, "R")
    assert result["LapTime (s)"].tolist() == [90.5, 91.2]


def test_fetch_weather_forecast_without_api_key_returns_neutral_defaults(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    weather = WeatherConfig(latitude=1.0, longitude=1.0, forecast_time="2025-01-01 00:00:00")

    rain, temp = loader.fetch_weather_forecast(weather)

    assert (rain, temp) == (0.0, 20.0)


def test_fetch_weather_forecast_parses_matching_slot():
    weather = WeatherConfig(latitude=1.0, longitude=1.0, forecast_time="2025-01-01 12:00:00")
    payload = {
        "list": [
            {"dt_txt": "2025-01-01 09:00:00", "pop": 0.1, "main": {"temp": 15.0}},
            {"dt_txt": "2025-01-01 12:00:00", "pop": 0.8, "main": {"temp": 22.5}},
        ]
    }
    fake_response = MagicMock()
    fake_response.json.return_value = payload
    fake_response.raise_for_status = MagicMock()

    with patch("f1_predictor.data.loader.requests.get", return_value=fake_response):
        rain, temp = loader.fetch_weather_forecast(weather, api_key="test-key")

    assert (rain, temp) == (0.8, 22.5)


def test_fetch_weather_forecast_falls_back_on_request_error():
    weather = WeatherConfig(latitude=1.0, longitude=1.0, forecast_time="2025-01-01 12:00:00")

    with patch("f1_predictor.data.loader.requests.get", side_effect=ConnectionError("boom")):
        rain, temp = loader.fetch_weather_forecast(weather, api_key="test-key")

    assert (rain, temp) == (0.0, 20.0)
