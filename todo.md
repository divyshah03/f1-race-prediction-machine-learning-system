# F1 Race Prediction — Modernization TODO

Goal: turn 9 duplicated per-race scripts into one clean, production-style ML project
that demonstrates real engineering + MLOps + deployment skills.

---

## Phase 0 — Setup (30 min)

- [x] ~~Create a new branch~~ — done directly on `main` instead (see note below)
- [x] Set up a proper project structure:
  ```
  f1-predictor/
    src/
      data/          # loading, caching, FastF1 wrapper
      features/       # feature engineering
      models/         # training, prediction
      pipeline.py      # ties it all together
    configs/
      races/           # one YAML per race (circuit info, quali times, weather)
    api/               # FastAPI app
    tests/
    notebooks/         # exploration only, not production code
    Dockerfile
    requirements.txt
    README.md
  ```
- [x] Set up a virtual environment + pin dependencies in `requirements.txt`
- [x] Init `pyproject.toml` so it's installable as a package (`pip install -e .`)

> Decision: skipped the separate `refactor/unified-pipeline` branch and committed the
> restructure directly to `main`, since it was requested as its own reviewable commit
> rather than a PR. The actual repo root wasn't renamed to `f1-predictor/` either — this
> repo is the user's working directory and renaming it would have broken the open
> session/IDE; the internal layout matches the target structure instead.

---

## Phase 1 — Refactor: kill the duplication (highest priority)

- [x] Diff the 9 race scripts and identify what's actually different between them
      (was: circuit name, quali times, weather coords/time, team/points snapshots,
      wet-performance/clean-air-pace reference dicts, and per-race model hyperparams)
- [x] Move race-specific data into `configs/races/<race_name>.yaml`
      (quali times, driver/team lineup, circuit metadata)
- [x] Write one shared `src/f1_predictor/data/loader.py`:
  - [x] FastF1 data pulling + local caching (single implementation)
  - [x] OpenWeatherMap fetch, with API key from `.env` (never hardcoded; was a
        literal `"YOURAPIKEY"` string in every script before)
- [x] Write one shared `src/f1_predictor/features/engineer.py`:
  - [x] Sector times, team performance score, clean-air pace, wet-weather factor
  - [x] Wrap preprocessing as a `sklearn.ColumnTransformer` (median-impute numeric
        features, one-hot the circuit), not loose per-race functions
- [x] Write one shared `src/f1_predictor/models/train.py` and `predict.py`
- [x] Write `src/f1_predictor/pipeline.py`: `run(race)` that does load → feature → train → predict
- [x] Delete the 9 old scripts once the unified pipeline reproduces their results
      (verified by running all 9 race configs end to end against live FastF1 data)
- [x] Add a `.env.example` and `.gitignore` (API keys, cache dir, `__pycache__`)

**Checkpoint:** you can run `python -m src.pipeline --race monaco` and get the same
output the old `monaco.py` gave you, from one shared codebase.

---

## Phase 2 — ML upgrades

- [x] Add `circuit` as a categorical feature and train **one unified model across
      all races** instead of 9 separate models (`src/f1_predictor/unified.py`)
- [x] Switch to **time-aware train/test splits** (walk-forward by historical round:
      train on early-season rounds, test on the latest ones) — no random splits
- [x] Add **CatBoost** and **LightGBM** as candidate models alongside XGBoost;
      benchmark all four (incl. GradientBoosting) with a consistent walk-forward split
- [x] Add **SHAP** for feature importance/explainability (`src/f1_predictor/explain.py`,
      replaces the default `feature_importances_` bar charts)
- [x] Reframe output: added Monte Carlo **podium probability per driver**
      (`models/predict.py::podium_probabilities`, perturbs the point prediction by
      the model's own held-out residual noise) alongside the point prediction
- [x] Add a simple baseline model ("predict quali order") to prove the model adds
      value over the naive baseline — report the lift
- [x] Write `evaluate.py` that reports MAE, RMSE, and rank-correlation (Spearman)
      between predicted and actual finishing order

**Real result** (walk-forward split, last 2 rounds held out — Monaco, Abu
Dhabi — all 9 races): gradient_boosting MAE 2.60s vs. baseline 8.01s
(**-67.6% MAE**). Spearman is averaged **per held-out race**, not pooled
across races (pooling conflated "ranks drivers correctly" with "separates two
circuits' pace" and briefly made catboost look badly broken at -0.33 — see
README Results for the full explanation); per-race, no model wins on both
axes: gradient_boosting 0.33 (worse than baseline's 0.50), catboost 0.54
(best of all four, on the worst MAE). An honest, non-cherry-picked finding.

**Checkpoint:** one trained model, benchmarked against 2-3 algorithms and a
naive baseline, with SHAP plots and a written note on what actually mattered.

---

## Phase 3 — Experiment tracking (MLflow)

- [x] `pip install mlflow`
- [x] Wrap training runs with `mlflow.start_run()` (`src/f1_predictor/tracking.py`), log:
  - [x] hyperparameters
  - [x] MAE / RMSE / rank-correlation
  - [x] model artifact — fixed via `skops_trusted_types=` (see
        `tracking.SKOPS_TRUSTED_TYPES`); verified by loading every candidate
        model type (gradient_boosting/xgboost/lightgbm/catboost) back from its
        logged MLflow artifact and confirming `.predict()` works.
  - [x] feature importance plot as an artifact — `unified.benchmark_models`
        now builds a SHAP summary figure per run and logs it via
        `mlflow.log_figure` as `shap_summary.png`
- [ ] Run `mlflow ui` locally and screenshot it for the README — needs a human at a
      browser; not something this session could do. Run `mlflow ui --backend-store-uri
      sqlite:///mlflow.db` yourself to see it (README Usage section has the command).
- [ ] (Optional stretch) log input data version/hash — not done

> Note: this MLflow install defaults its tracking URI to a local sqlite file
> (`mlflow.db`) rather than the classic `./mlruns/` folder. In this repo's path
> (containing an apostrophe) that URI got mis-encoded and the DB file was never
> actually created — runs printed correctly but didn't persist. Set
> `MLFLOW_TRACKING_URI` explicitly in `.env` (e.g. to an absolute path with no
> special characters) to get a real local store.

**Checkpoint:** you can pull up MLflow and show a table comparing every model
run you've done, not just the final one.

---

## Phase 4 — Testing + CI

- [x] Write unit tests (`pytest`) for:
  - [x] feature engineering functions (given known input, expect known output)
  - [x] data loader (mock the FastF1/weather API calls)
  - [x] pipeline runs end-to-end on a small fixture race
  - [x] (also: evaluate.py metrics, baseline model, API endpoints) — 19 tests, all green
- [x] Add `ruff` + `black` for linting/formatting (both configured in `pyproject.toml`,
      both currently clean on `src`/`api`/`tests`)
- [x] Add `.github/workflows/ci.yml`: run tests + lint on every push/PR — pushed to
      `origin/main` and green (fixed a real CI-only failure: `httpx` was only ever
      installed locally as a transitive dep of the optional `llm` extra, so CI's
      `pip install ".[dev]"` crashed FastAPI's `TestClient` import; added `httpx` to
      `dev` deps explicitly)
- [x] Add a CI badge to the README

**Checkpoint:** green CI badge on your repo — signals real software practice,
not just notebooks.

---

## Phase 5 — Deployment

- [x] Write a `Dockerfile` that installs deps and runs the API
- [x] Build a **FastAPI** app (`api/main.py`): `POST /predict` takes `{"race": "..."}`
      and returns predicted order, podium probabilities, MAE/Spearman lift over
      baseline, and an optional LLM summary; `GET /races` lists valid race slugs
- [x] Test the container locally (`docker build . && docker run ...`) — built the
      image, ran it, and hit `/health`, `/races`, and `/predict` against real
      (non-mocked) FastF1 + weather data. Surfaced and fixed a real container-only
      bug: `config.CONFIGS_ROOT` was computed from `Path(__file__).resolve().parents[2]`,
      which only lands on the repo root under an *editable* install; the Docker
      image's regular `pip install ".[api]"` copies `config.py` into site-packages
      instead, so `/races` came back empty and every `/predict` 404'd. Fixed with an
      `F1_CONFIGS_DIR` env var override (same pattern as `loader.py`'s `F1_CACHE_DIR`),
      set in the Dockerfile.
- [ ] Deploy to a free-tier host (Render, Railway, Fly.io) — **intentionally not
      done**: the user has explicitly said they do not want this project deployed
      anywhere. The Dockerfile + API are verified working locally if that changes.
- [ ] (Optional) Streamlit front end — not built (FastAPI response already carries
      everything a UI would need)
- [ ] Get a live public URL — not pursued, per the user's no-deploy preference above.

**Checkpoint:** a link you can click that shows a working prediction, not just
a GitHub repo someone has to clone to try.

---

## Phase 6 — LLM reasoning layer (differentiator, optional but high value)

- [x] Take the SHAP values for a given race's prediction (`explain.py::explain_prediction`)
- [x] Write a prompt template that turns them into a natural-language summary
      (`src/f1_predictor/llm_summary.py`)
- [x] Call an LLM API (Claude, via the `anthropic` SDK) to generate this summary
- [x] Surface it in the API response (`POST /predict` with `"include_llm_summary": true`)
- [x] Note in README: demonstrates LLM-integration skills on top of classic ML

> Requires `ANTHROPIC_API_KEY` in `.env`; gracefully returns `None` (skips the
> feature) if unset rather than erroring, since this is meant as an optional
> add-on, not a dependency of the core pipeline. Still not actually exercised
> against a live API key — offered to the user this session (paste a key, add
> it to `.env` themselves, or skip), and they chose to skip it. The code path
> is verified to import and wire correctly; the real Claude call itself
> remains untested.

---

## Phase 7 — Polish for resume/portfolio

- [x] Rewrite the README:
  - [x] Problem statement + why it's non-trivial (F1 variance, DNFs, strategy)
  - [x] Architecture diagram (data → features → model → API/UI)
  - [x] Results: model comparison table, baseline lift, honest read of the numbers
  - [ ] Live demo link + screenshot/gif — blocked on Phase 5 deployment
  - [x] "What I'd do with more time" section
- [x] Write resume bullets (in README's "Resume bullets" section)
- [ ] Prepare a 60-90 second verbal walkthrough for interviews — this is a "you,
      out loud, in an interview" exercise, not something to draft as a script here;
      the README's "Why this is harder than it looks" + "Results" sections have the
      raw material for one.

---

## Suggested order if short on time

If you can't do everything, prioritize in this order for maximum resume/interview impact:
1. Phase 1 (refactor) — fixes the biggest red flag
2. Phase 2 (ML upgrades, at least unified model + time-aware split + baseline)
3. Phase 5 (deployment) — gives you a live demo link
4. Phase 3 (MLflow) — cheap to add, strong signal
5. Phase 4 (CI/tests) — cheap, strong signal
6. Phase 6 (LLM layer) — differentiator if time allows
7. Phase 7 (polish) — always do this last, right before applying