import pandas as pd
from fastapi.testclient import TestClient

from api.main import _cached_run, app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_races_endpoint_lists_configs():
    client = TestClient(app)
    response = client.get("/races")
    assert response.status_code == 200
    assert "monaco_gp" in response.json()


def test_predict_endpoint_returns_prediction(monkeypatch):
    fake_outcome = {
        "config": type("Cfg", (), {"name": "Fixture Grand Prix"})(),
        "results": pd.DataFrame(
            {"Driver": ["VER", "NOR", "LEC"], "PredictedRaceTime (s)": [90.0, 91.0, 92.0]}
        ),
        "model_metrics": {"mae": 1.0, "rmse": 1.2, "spearman": 0.8},
        "baseline_metrics": {"mae": 1.5, "rmse": 1.8, "spearman": 0.6},
        "lift": {"spearman_lift": 0.2, "mae_improvement_pct": 33.3},
    }

    _cached_run.cache_clear()
    monkeypatch.setattr("api.main.run_pipeline", lambda race: fake_outcome)

    client = TestClient(app)
    response = client.post("/predict", json={"race": "fixture_gp"})

    assert response.status_code == 200
    body = response.json()
    assert body["podium"] == ["VER", "NOR", "LEC"]
    assert body["model_mae_s"] == 1.0
