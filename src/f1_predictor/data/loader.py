"""Data loading: FastF1 historical sessions and OpenWeatherMap forecasts.

This module is the single implementation shared by every race, replacing the
nine copy-pasted ``initialize_cache`` / ``load_historical_race_data`` /
``fetch_weather_data`` functions that used to live in ``Races/*.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import fastf1
import pandas as pd
import requests

from f1_predictor.config import HistoricalSessionConfig, WeatherConfig

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.getenv("F1_CACHE_DIR", "f1_cache"))
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"

_cache_initialized = False


def enable_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Enable FastF1's on-disk cache (idempotent)."""
    global _cache_initialized
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    _cache_initialized = True


def load_historical_laps(historical: HistoricalSessionConfig) -> pd.DataFrame:
    """Load lap + sector time data (in seconds) for a historical session."""
    if not _cache_initialized:
        enable_cache()

    session = fastf1.get_session(historical.year, historical.round, historical.session)
    session.load()

    columns = ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    laps = session.laps[columns].copy()
    laps.dropna(inplace=True)

    for column in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{column} (s)"] = laps[column].dt.total_seconds()

    return laps


def average_sector_times(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-driver mean sector times plus a derived total sector time."""
    sector_cols = ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    if laps.empty:
        return pd.DataFrame(columns=["Driver", *sector_cols, "TotalSectorTime (s)"])

    sectors = laps.groupby("Driver")[sector_cols].mean().reset_index()
    sectors["TotalSectorTime (s)"] = sectors[sector_cols].sum(axis=1)
    return sectors


def fetch_weather_forecast(weather: WeatherConfig, api_key: str | None = None) -> tuple[float, float]:
    """Return (rain_probability, temperature_celsius) for the configured forecast slot.

    Falls back to a neutral (0.0, 20.0) forecast if no API key is configured or the
    request fails, so the pipeline stays runnable offline / without credentials.
    """
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.info("OPENWEATHER_API_KEY not set; using neutral weather defaults.")
        return 0.0, 20.0

    params = {
        "lat": weather.latitude,
        "lon": weather.longitude,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        forecast = next(
            (f for f in payload["list"] if f["dt_txt"] == weather.forecast_time),
            None,
        )
        if forecast is None:
            return 0.0, 20.0
        return forecast["pop"], forecast["main"]["temp"]
    except (requests.RequestException, KeyError, ValueError):
        logger.warning("Weather fetch failed; using neutral weather defaults.", exc_info=True)
        return 0.0, 20.0
