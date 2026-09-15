"""FastAPI service exposing the F1 race prediction pipeline over HTTP.

Run locally with:
    uvicorn api.main:app --reload

Endpoints:
    GET  /races            -> list available race config slugs
    POST /predict           -> run the pipeline for a race and return predictions
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from f1_predictor.config import available_races
from f1_predictor.pipeline import run as run_pipeline

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predicts F1 race finishing order from qualifying, historical, and weather data.",
    version="0.1.0",
)


class DriverPrediction(BaseModel):
    driver: str
    predicted_race_time_s: float


class PredictResponse(BaseModel):
    race: str
    predicted_order: list[DriverPrediction]
    podium: list[str]
    model_mae_s: float
    model_spearman: float
    baseline_spearman: float
    spearman_lift: float


class PredictRequest(BaseModel):
    race: str


@app.get("/races")
def list_races() -> list[str]:
    return available_races()


@lru_cache(maxsize=32)
def _cached_run(race: str) -> dict:
    return run_pipeline(race)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        outcome = _cached_run(request.race)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    results = outcome["results"]
    return PredictResponse(
        race=outcome["config"].name,
        predicted_order=[
            DriverPrediction(driver=row["Driver"], predicted_race_time_s=row["PredictedRaceTime (s)"])
            for _, row in results.iterrows()
        ],
        podium=list(results["Driver"].head(3)),
        model_mae_s=outcome["model_metrics"]["mae"],
        model_spearman=outcome["model_metrics"]["spearman"],
        baseline_spearman=outcome["baseline_metrics"]["spearman"],
        spearman_lift=outcome["lift"]["spearman_lift"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
