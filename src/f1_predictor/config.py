"""Race configuration loading from configs/races/*.yaml."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import yaml


def _default_configs_root() -> Path:
    """Locate configs/races/, overridable via F1_CONFIGS_DIR (see .env.example).

    `Path(__file__).resolve().parents[2]` only lands on the repo root for an
    *editable* install (`pip install -e .`), where this file still lives under
    `src/f1_predictor/`. A regular install (e.g. the Dockerfile's `pip install
    ".[api]"`) copies this module into site-packages, several directories away
    from any `configs/` folder, so that guess silently resolves to a path that
    doesn't exist and `available_races()` returns an empty list. F1_CONFIGS_DIR
    lets a packaged deployment point at wherever it actually copied configs/.
    """
    env_dir = os.getenv("F1_CONFIGS_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parents[2] / "configs" / "races"


CONFIGS_ROOT = _default_configs_root()


@dataclasses.dataclass
class HistoricalSessionConfig:
    year: int
    round: int
    session: str = "R"


@dataclasses.dataclass
class WeatherConfig:
    latitude: float
    longitude: float
    forecast_time: str
    rain_threshold: float = 0.75


@dataclasses.dataclass
class ModelConfig:
    type: str = "gradient_boosting"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int | None = None
    test_size: float = 0.2
    random_state: int = 39
    monotone_constraints: str | None = None


@dataclasses.dataclass
class DriverEntry:
    code: str
    qualifying_time: float | None


@dataclasses.dataclass
class RaceConfig:
    name: str
    circuit: str
    season: int
    historical: HistoricalSessionConfig
    weather: WeatherConfig
    model: ModelConfig
    drivers: list[DriverEntry]
    driver_team: dict[str, str] = dataclasses.field(default_factory=dict)
    team_points: dict[str, float] = dataclasses.field(default_factory=dict)
    clean_air_race_pace: dict[str, float] = dataclasses.field(default_factory=dict)
    wet_performance_factor: dict[str, float] = dataclasses.field(default_factory=dict)
    season_points: dict[str, float] = dataclasses.field(default_factory=dict)
    average_2025_performance: dict[str, float] = dataclasses.field(default_factory=dict)
    average_position_change: dict[str, float] = dataclasses.field(default_factory=dict)
    last_year_winner: str | None = None
    qualifying_time_transform: str = "none"  # "none" | "square"

    @property
    def slug(self) -> str:
        return self.circuit


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def available_races() -> list[str]:
    """List race config slugs (filenames without extension) under configs/races/."""
    return sorted(p.stem for p in CONFIGS_ROOT.glob("*.yaml"))


def load_race_config(race: str | Path) -> RaceConfig:
    """Load a RaceConfig by slug (e.g. 'monaco_gp') or by explicit path to a YAML file."""
    path = Path(race)
    if not path.suffix:
        path = CONFIGS_ROOT / f"{race}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No race config found for '{race}' (looked at {path})")

    raw = _load_yaml(path)

    historical = HistoricalSessionConfig(**raw["historical"])
    weather = WeatherConfig(**raw["weather"])
    model = ModelConfig(**raw.get("model", {}))
    drivers = [DriverEntry(**d) for d in raw["drivers"]]

    return RaceConfig(
        name=raw["name"],
        circuit=raw["circuit"],
        season=raw["season"],
        historical=historical,
        weather=weather,
        model=model,
        drivers=drivers,
        driver_team=raw.get("driver_team", {}),
        team_points=raw.get("team_points", {}),
        clean_air_race_pace=raw.get("clean_air_race_pace", {}),
        wet_performance_factor=raw.get("wet_performance_factor", {}),
        season_points=raw.get("season_points", {}),
        average_2025_performance=raw.get("average_2025_performance", {}),
        average_position_change=raw.get("average_position_change", {}),
        last_year_winner=raw.get("last_year_winner"),
        qualifying_time_transform=raw.get("qualifying_time_transform", "none"),
    )
