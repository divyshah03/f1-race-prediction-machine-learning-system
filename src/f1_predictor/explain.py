"""SHAP-based feature importance and per-prediction explainability.

Replaces the plain ``model.feature_importances_`` bar charts every race script
used to draw with SHAP, which explains individual predictions rather than just
global feature ranking (Phase 2 of todo.md).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _transform_and_names(pipeline: Pipeline, table: pd.DataFrame, columns: list[str]):
    preprocessing = pipeline.named_steps["preprocessing"]
    transformed = preprocessing.transform(table[columns])
    feature_names = preprocessing.get_feature_names_out()
    return transformed, feature_names


def shap_summary_plot(pipeline: Pipeline, table: pd.DataFrame, columns: list[str], output_path: str) -> None:
    """Save a SHAP summary (beeswarm) plot for the model step of `pipeline`."""
    model = pipeline.named_steps["model"]
    transformed, feature_names = _transform_and_names(pipeline, table, columns)

    explainer = shap.Explainer(model, transformed)
    shap_values = explainer(transformed)

    plt.figure()
    shap.summary_plot(shap_values, features=transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def explain_prediction(pipeline: Pipeline, table: pd.DataFrame, columns: list[str], driver: str) -> dict[str, float]:
    """Per-driver SHAP feature contributions, used to feed the optional LLM summary."""
    model = pipeline.named_steps["model"]
    row = table[table["Driver"] == driver]
    transformed, feature_names = _transform_and_names(pipeline, row, columns)

    explainer = shap.Explainer(model, transformed)
    shap_values = explainer(transformed)

    return dict(zip(feature_names, shap_values.values[0]))
