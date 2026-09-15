# F1 Predictor

A machine learning pipeline that predicts Formula 1 race outcomes from qualifying
times, historical race telemetry (via [FastF1](https://github.com/theOehrly/Fast-F1)),
and race-day weather forecasts.

This project started as nine near-identical, copy-pasted per-race scripts. It has
been refactored into a single configurable pipeline: one loader, one feature
engineering module, one model-training module, driven by a small YAML config per race.

## Project Structure

```
f1-predictor/
├── src/f1_predictor/
│   ├── config.py           # loads configs/races/*.yaml into typed dataclasses
│   ├── data/
│   │   └── loader.py       # FastF1 session loading + caching, OpenWeatherMap fetch
│   ├── features/
│   │   └── engineer.py     # qualifying + sector + weather + reference data -> feature table
│   ├── models/
│   │   ├── train.py        # sklearn Pipeline: ColumnTransformer(impute) + regressor
│   │   ├── predict.py       # score a fitted pipeline, rank drivers
│   │   └── baseline.py      # naive "predict quali order" baseline
│   ├── evaluate.py          # MAE / RMSE / Spearman rank-correlation
│   ├── pipeline.py          # single-race CLI: load -> feature -> train -> predict -> evaluate
│   ├── unified.py           # cross-race model, walk-forward split, model benchmark
│   ├── explain.py           # SHAP feature importance / per-prediction explanations
│   ├── tracking.py          # MLflow experiment logging
│   └── llm_summary.py       # optional: SHAP -> natural-language race summary (Claude)
├── configs/races/*.yaml     # one file per Grand Prix: quali times, weather, teams, model hyperparams
├── api/main.py               # FastAPI service (`/predict`, `/races`)
├── tests/                    # pytest unit + pipeline tests, FastF1/weather calls mocked
├── notebooks/                 # exploration only, never production code
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[api,llm,dev]"
cp .env.example .env   # fill in OPENWEATHER_API_KEY / ANTHROPIC_API_KEY if you have them
```

## Usage

Run the pipeline for one race:

```bash
python -m f1_predictor.pipeline --race monaco_gp
```

Available race slugs live in `configs/races/` (e.g. `bahrain_gp`, `monaco_gp`,
`abu_dhabi_gp`). Each prints the predicted finishing order, model MAE/RMSE/Spearman,
how that compares to the naive "predict qualifying order" baseline, and the
predicted podium.

Train and benchmark a single model across every race (Phase 2):

```bash
python -m f1_predictor.unified
```

Serve predictions over HTTP:

```bash
uvicorn api.main:app --reload
# then: curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"race": "monaco_gp"}'
```

Run tests:

```bash
pytest
ruff check src api tests
black --check src api tests
```

## Adding a race

Add a new `configs/races/<slug>.yaml` (see any existing file for the schema:
`historical` round/year/session, `weather` coordinates + forecast time, `model`
hyperparameters, `drivers` with qualifying times, and any optional reference data
like `driver_team` / `team_points` / `clean_air_race_pace`). No code changes needed.

## What's next

See `todo.md` for the in-progress modernization roadmap (unified model, MLflow
tracking, CI, Docker/FastAPI deployment, SHAP explainability, optional LLM summary
layer).
