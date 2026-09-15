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
from f1_predictor.explain import explain_prediction
from f1_predictor.llm_summary import summarize_prediction
from f1_predictor.pipeline import run as run_pipeline

app = FastAPI(
    title="F1 Race Predictor API",
    description="Predicts F1 race finishing order from qualifying, historical, and weather data.",
    version="0.1.0",
)


class DriverPrediction(BaseModel):
    driver: str
    predicted_race_time_s: float


class PodiumProbability(BaseModel):
    driver: str
    podium_probability: float
    win_probability: float


class PredictResponse(BaseModel):
    race: str
    predicted_order: list[DriverPrediction]
    podium: list[str]
    podium_probabilities: list[PodiumProbability]
    model_mae_s: float
    model_spearman: float
    baseline_spearman: float
    spearman_lift: float
    llm_summary: str | None = None


class PredictRequest(BaseModel):
    race: str
    include_llm_summary: bool = False


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
    probabilities = outcome["podium_probabilities"]

    llm_summary = None
    if request.include_llm_summary:
        winner = results.iloc[0]["Driver"]
        winner_shap = explain_prediction(outcome["pipeline"], results, outcome["columns"], winner)
        podium_for_prompt = [
            (row["Driver"], row["PredictedRaceTime (s)"]) for _, row in results.head(3).iterrows()
        ]
        llm_summary = summarize_prediction(outcome["config"].name, podium_for_prompt, winner_shap)

    return PredictResponse(
        race=outcome["config"].name,
        predicted_order=[
            DriverPrediction(driver=row["Driver"], predicted_race_time_s=row["PredictedRaceTime (s)"])
            for _, row in results.iterrows()
        ],
        podium=list(results["Driver"].head(3)),
        podium_probabilities=[
            PodiumProbability(
                driver=row["Driver"],
                podium_probability=row["PodiumProbability"],
                win_probability=row["WinProbability"],
            )
            for _, row in probabilities.iterrows()
        ],
        model_mae_s=outcome["model_metrics"]["mae"],
        model_spearman=outcome["model_metrics"]["spearman"],
        baseline_spearman=outcome["baseline_metrics"]["spearman"],
        spearman_lift=outcome["lift"]["spearman_lift"],
        llm_summary=llm_summary,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
