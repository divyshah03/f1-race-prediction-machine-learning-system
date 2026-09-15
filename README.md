# F1 Predictor

[![CI](https://github.com/divyshah03/f1-race-prediction-machine-learning-system/actions/workflows/ci.yml/badge.svg)](https://github.com/divyshah03/f1-race-prediction-machine-learning-system/actions/workflows/ci.yml)

A machine learning pipeline that predicts Formula 1 race outcomes from qualifying
times, historical race telemetry (via [FastF1](https://github.com/theOehrly/Fast-F1)),
and race-day weather forecasts.

## Why this is harder than it looks

Predicting F1 finishing order from qualifying isn't just "sort by grid position":
race pace diverges from qualifying pace, weather can flip the whole order, DNFs and
strategy calls add variance no lap-time model captures, and every circuit rewards a
different car/driver skill mix. This project doesn't solve strategy or DNFs — it
predicts *pace-based* finishing order from qualifying + historical + weather signals,
and is explicit (see Results below) about where that ceiling is.

This project started as nine near-identical, copy-pasted per-race scripts, each with
its own hardcoded API key, its own slightly-different feature engineering, and its
own bespoke plotting code. It's been refactored into a single configurable pipeline:
one loader, one feature engineering module, one model-training module, driven by a
small YAML config per race.

```
FastF1 (historical laps/sectors)  ─┐
OpenWeatherMap (race-day forecast) ─┼──►  feature table  ──►  model pipeline  ──►  predicted order
configs/races/<slug>.yaml (quali,  │      (engineer.py)     (ColumnTransformer      + podium probability
  teams, reference data)          ─┘                          + regressor)          + optional LLM summary
```

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

Every run of the above logs params, metrics, the fitted model (loadable back
via `mlflow.sklearn.load_model`, for every candidate type including
XGBoost/LightGBM/CatBoost), and a SHAP summary plot to MLflow. Inspect with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Serve predictions over HTTP:

```bash
uvicorn api.main:app --reload
# then: curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"race": "monaco_gp"}'
```

Or run it in Docker (verified end-to-end: builds, serves `/health`, `/races`,
and `/predict` with real, non-mocked FastF1 + weather data):

```bash
docker build -t f1-predictor .
docker run -p 8000:8000 f1-predictor
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

## Results

One `gradient_boosting` model trained across all 9 races (`circuit` as a one-hot
categorical feature), evaluated with a **walk-forward split** — trained on the
earlier rounds of the season, tested on the last 2 held-out rounds (Monaco, Abu
Dhabi) — not a random split, which would leak future races into training:

| Model                  | MAE (s) | RMSE (s) | Spearman (per-race) |
|-------------------------|--------:|---------:|---------:|
| **gradient_boosting**   | **2.60**| **3.61** | 0.33     |
| xgboost                 | 3.30    | 4.46     | 0.16     |
| lightgbm                | 5.92    | 6.94     | 0.26     |
| catboost                | 7.39    | 9.39     | **0.54**|
| baseline (quali order)  | 8.01    | 8.06     | 0.50     |

Reproduce with `python -m f1_predictor.unified`.

Spearman here is **averaged within each held-out race, then across races** —
not pooled across every test row. Rank correlation only means something within
a single race's field of drivers (nobody's finishing position is compared
against a driver from a different Grand Prix), so pooling Monaco and Abu Dhabi
rows together before computing one correlation would conflate "ranks drivers
correctly" with "happens to separate two circuits' overall pace" — an earlier
version of this table did exactly that and reported CatBoost at **-0.33**,
which looked like a modeling bug. It wasn't: CatBoost's model artifact,
categorical handling, and loss/eval setup all check out, and per-race it's
actually the *best*-ranking model of the four, ahead of the naive baseline.

**Honest read of this table:** no single model wins on both axes.
`gradient_boosting` has the lowest absolute error (**-67.6% MAE vs. baseline**)
but the worst rank-correlation lift of the boosted models — it's actually
*less* rank-correlated with the true order than just sorting by qualifying
time on this small held-out set. `catboost` is the opposite: worst MAE, best
ranking. That tension, not a clean sweep, is the real finding: qualifying
order is already a strong proxy for finishing order, so a model can shave
absolute-time error without necessarily reordering who beats whom, and vice
versa. (Also fixed along the way: XGBoost defaults to
`enable_categorical=True` in this xgboost version, which silently made SHAP's
`TreeExplainer` refuse to run — disabled explicitly, since `circuit` is
already one-hot encoded upstream and XGBoost never sees a native categorical
column.)

Beyond the point prediction, every race also gets Monte Carlo **podium/win
probabilities** (`models/predict.py::podium_probabilities`) by perturbing the
predicted time with the model's own held-out residual noise — e.g. for Monaco:
LEC 85% podium / 53% win, NOR 52%/15%, PIA 40%/9%.

## What I'd do with more time

- **Tune per-model-type**, not one generic hyperparameter set for all four
  candidates — `gradient_boosting` currently has the best MAE but the worst
  rank-correlation lift of the boosted models; a real tuning pass (not just a
  metric fix) might close that gap.
- **Pit stop strategy and tire degradation** — currently zero strategy signal;
  this is likely the single biggest accuracy ceiling.
- **DNF modeling** — races end early for reasons pace data can't see.
- **More seasons of history** — 9 races x ~13-20 drivers is a small dataset;
  the podium-probability confidence intervals are almost certainly too tight.
- **Exercise the LLM summary against a live API call** — the SHAP-to-natural-
  language path (`llm_summary.py`) is wired into the API and imports/runs
  correctly, but generating and saving a real example output needs an
  `ANTHROPIC_API_KEY`, which wasn't provided this session (offered and
  deliberately skipped).
- **Actually deploy it** — Dockerfile + FastAPI are verified working end-to-end
  locally (built the image, ran the container, hit `/health`, `/races`, and
  `/predict` against live, non-mocked FastF1 data); getting a public URL on
  Render/Railway/Fly.io is a deliberate manual step, not attempted here.

## Resume bullets

- Refactored a 9-script, copy-pasted F1 prediction codebase (hardcoded API keys,
  duplicated feature logic) into one configurable pipeline driven by per-race YAML,
  cutting the codebase to a single implementation per concern.
- Benchmarked GradientBoosting/XGBoost/LightGBM/CatBoost with a time-aware
  (walk-forward) split against a naive baseline, cutting MAE ~68% and adding
  SHAP-based explainability and Monte Carlo podium probabilities.
- Built a FastAPI prediction service (Dockerized, tested with pytest + CI via
  GitHub Actions) with an optional LLM layer that turns SHAP values into a
  natural-language race explanation.

## What's next

See `todo.md` for the full in-progress modernization roadmap and an honest,
checkbox-level account of what's done vs. still open (mainly: live deployment,
a real LLM-summary example from a live API key, and per-model hyperparameter
tuning).
