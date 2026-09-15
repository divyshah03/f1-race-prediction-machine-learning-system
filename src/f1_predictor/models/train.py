"""Model training: builds a preprocessing + regressor sklearn Pipeline.

Supports GradientBoosting (scikit-learn) and XGBoost/LightGBM/CatBoost as
interchangeable candidate regressors, selected per race via ``model.type``
in the race's YAML config.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from f1_predictor.config import ModelConfig
from f1_predictor.features.engineer import build_preprocessing_pipeline, numeric_feature_columns


def _make_regressor(model_config: ModelConfig):
    kwargs: dict = dict(
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        random_state=model_config.random_state,
    )
    if model_config.max_depth is not None:
        kwargs["max_depth"] = model_config.max_depth

    if model_config.type == "gradient_boosting":
        return GradientBoostingRegressor(**kwargs)

    if model_config.type == "xgboost":
        from xgboost import XGBRegressor

        if model_config.monotone_constraints:
            kwargs["monotone_constraints"] = model_config.monotone_constraints
        return XGBRegressor(**kwargs)

    if model_config.type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(verbosity=-1, **kwargs)

    if model_config.type == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(verbose=False, **kwargs)

    raise ValueError(f"Unknown model type: {model_config.type!r}")


def build_model_pipeline(
    model_config: ModelConfig, numeric_features: list[str], include_circuit: bool = False
) -> Pipeline:
    preprocessing = build_preprocessing_pipeline(numeric_features, include_circuit=include_circuit)
    regressor = _make_regressor(model_config)
    return Pipeline([("preprocessing", preprocessing), ("model", regressor)])


def fit(
    table: pd.DataFrame, model_config: ModelConfig, include_circuit: bool = False
) -> tuple[Pipeline, list[str]]:
    """Fit a model pipeline on a feature table that includes the 'LapTime (s)' target.

    Returns the fitted pipeline and the exact input column order it expects.
    """
    features = numeric_feature_columns(table)
    columns = features + (["circuit"] if include_circuit else [])
    pipeline = build_model_pipeline(model_config, features, include_circuit=include_circuit)
    pipeline.fit(table[columns], table["LapTime (s)"])
    return pipeline, columns
