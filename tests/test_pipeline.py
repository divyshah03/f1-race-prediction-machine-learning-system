import yaml

from f1_predictor import pipeline


def test_pipeline_run_end_to_end_on_fixture(monkeypatch, sample_race_config, sample_historical_laps, tmp_path):
    config_path = tmp_path / "fixture_gp.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "name": sample_race_config.name,
                "circuit": sample_race_config.circuit,
                "season": sample_race_config.season,
                "historical": {"year": 2024, "round": 1, "session": "R"},
                "weather": {"latitude": 0.0, "longitude": 0.0, "forecast_time": "2025-01-01 00:00:00"},
                "model": {
                    "type": "gradient_boosting",
                    "n_estimators": 10,
                    "learning_rate": 0.5,
                    "test_size": 0.34,
                    "random_state": 1,
                },
                "drivers": [
                    {"code": "VER", "qualifying_time": 80.0},
                    {"code": "NOR", "qualifying_time": 81.0},
                    {"code": "LEC", "qualifying_time": 82.0},
                ],
            }
        )
    )

    monkeypatch.setattr(pipeline.loader, "load_historical_laps", lambda historical: sample_historical_laps)
    monkeypatch.setattr(pipeline.loader, "fetch_weather_forecast", lambda weather: (0.0, 20.0))

    outcome = pipeline.run(str(config_path))

    assert set(outcome["results"]["Driver"]) == {"VER", "NOR", "LEC"}
    assert "mae" in outcome["model_metrics"]
    assert "spearman" in outcome["baseline_metrics"]
    assert outcome["config"].name == "Fixture Grand Prix"
