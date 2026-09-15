# F1 Race Prediction — Modernization TODO

Goal: turn 9 duplicated per-race scripts into one clean, production-style ML project
that demonstrates real engineering + MLOps + deployment skills.

---

## Phase 0 — Setup (30 min)

- [ ] Create a new branch: `git checkout -b refactor/unified-pipeline`
- [ ] Set up a proper project structure:
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
- [ ] Set up a virtual environment + pin dependencies in `requirements.txt`
- [ ] Init `pyproject.toml` or `setup.py` so it's installable as a package

---

## Phase 1 — Refactor: kill the duplication (highest priority)

- [ ] Diff the 9 race scripts and identify what's actually different between them
      (should mostly be: circuit name, quali times, weather inputs)
- [ ] Move race-specific data into `configs/races/<race_name>.yaml`
      (quali times, driver/team lineup, circuit metadata)
- [ ] Write one shared `src/data/loader.py`:
  - [ ] FastF1 data pulling + local caching (single implementation)
  - [ ] OpenWeatherMap fetch, with API key from `.env` (never hardcoded)
- [ ] Write one shared `src/features/engineer.py`:
  - [ ] Sector times, team performance score, clean-air pace, wet-weather factor
  - [ ] Wrap as an `sklearn.Pipeline` / `ColumnTransformer`, not loose functions
- [ ] Write one shared `src/models/train.py` and `predict.py`
- [ ] Write `src/pipeline.py`: `run(race_config_path)` that does load → feature → train → predict
- [ ] Delete the 9 old scripts once the unified pipeline reproduces their results
- [ ] Add a `.env.example` and `.gitignore` (API keys, cache dir, `__pycache__`)

**Checkpoint:** you can run `python -m src.pipeline --race monaco` and get the same
output the old `monaco.py` gave you, from one shared codebase.

---

## Phase 2 — ML upgrades

- [ ] Add `circuit` / `race_name` as a categorical feature and train **one unified
      model across all races/seasons** instead of 9 separate models
- [ ] Switch to **time-aware train/test splits** (walk-forward validation across
      the season) — no random splits on race data
- [ ] Add **CatBoost** and **LightGBM** as candidate models alongside XGBoost;
      benchmark all three with consistent CV
- [ ] Add **SHAP** for feature importance/explainability (replace default
      `feature_importances_` plots)
- [ ] Reframe output: instead of only point-predicting lap time, add either
      - [ ] predicted finishing position distribution, or
      - [ ] podium probability per driver
- [ ] Add a simple baseline model (e.g. "predict quali order") to prove your
      model actually adds value over the naive baseline — report the lift
- [ ] Write a `evaluate.py` that reports MAE, RMSE, and rank-correlation
      (Spearman) between predicted and actual finishing order

**Checkpoint:** one trained model, benchmarked against 2-3 algorithms and a
naive baseline, with SHAP plots and a written note on what actually mattered.

---

## Phase 3 — Experiment tracking (MLflow)

- [ ] `pip install mlflow`
- [ ] Wrap training runs with `mlflow.start_run()`, log:
  - [ ] hyperparameters
  - [ ] MAE / RMSE / rank-correlation
  - [ ] model artifact
  - [ ] feature importance plot as an artifact
- [ ] Run `mlflow ui` locally and screenshot a comparison of your model runs
      for your README
- [ ] (Optional stretch) log input data version/hash so runs are reproducible

**Checkpoint:** you can pull up MLflow and show a table comparing every model
run you've done, not just the final one.

---

## Phase 4 — Testing + CI

- [ ] Write unit tests (`pytest`) for:
  - [ ] feature engineering functions (given known input, expect known output)
  - [ ] data loader (mock the FastF1/weather API calls)
  - [ ] pipeline runs end-to-end on a small fixture race
- [ ] Add `ruff` or `flake8` + `black` for linting/formatting
- [ ] Add `.github/workflows/ci.yml`: run tests + lint on every push/PR
- [ ] Add a badge to your README showing CI status

**Checkpoint:** green CI badge on your repo — signals real software practice,
not just notebooks.

---

## Phase 5 — Deployment

- [ ] Write a `Dockerfile` that installs deps and runs the pipeline/API
- [ ] Build a **FastAPI** app (`api/main.py`) with an endpoint like:
      `POST /predict` → takes race + driver inputs, returns predicted order
- [ ] Test the container locally: `docker build . && docker run ...`
- [ ] Deploy to a free-tier host (Render, Railway, or Fly.io)
- [ ] (Optional, easier demo) build a small **Streamlit** front end that hits
      the API and shows predicted podium + SHAP explanation, deploy on
      Streamlit Community Cloud
- [ ] Get a live public URL — put it at the top of your README

**Checkpoint:** a link you can click that shows a working prediction, not just
a GitHub repo someone has to clone to try.

---

## Phase 6 — LLM reasoning layer (differentiator, optional but high value)

- [ ] Take the SHAP values for a given race's prediction
- [ ] Write a prompt template that turns them into a natural-language summary:
      "Model predicts Verstappen P1 primarily due to clean-air pace and low
      wet-weather risk factor..."
- [ ] Call an LLM API (Anthropic/OpenAI) to generate this summary
- [ ] Surface it in the Streamlit UI or API response
- [ ] Note in README: this demonstrates RAG-adjacent / LLM-integration skills
      on top of classic ML — this is currently one of the most requested combo
      skill sets in ML job postings

---

## Phase 7 — Polish for resume/portfolio

- [ ] Rewrite the README:
  - [ ] Problem statement + why it's non-trivial (F1 variance, DNFs, strategy)
  - [ ] Architecture diagram (data → features → model → API/UI)
  - [ ] Results: model comparison table, baseline lift, MAE trend across season
  - [ ] Live demo link + screenshot/gif
  - [ ] "What I'd do with more time" section (pit strategy, tire degradation,
        multi-race ensembling — keep these as honest future work, don't overclaim)
- [ ] Write 2-3 resume bullets from this, e.g.:
  - "Refactored a 9-script ML codebase into a single configurable pipeline,
    reducing code duplication by ~80%"
  - "Built and deployed a FastAPI prediction service with Docker, MLflow
    experiment tracking, and CI"
  - "Benchmarked XGBoost/LightGBM/CatBoost with time-aware validation,
    improving rank-correlation over a naive baseline by X%"
- [ ] Prepare a 60-90 second verbal walkthrough for interviews (what it does,
      what you improved, what you'd do next)

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