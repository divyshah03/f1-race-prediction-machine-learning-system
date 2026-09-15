"""Naive baseline: predict finishing order = qualifying order (no ML at all).

Used to prove the trained model actually adds value over "just look at qualifying".
"""

from __future__ import annotations

import pandas as pd


def baseline_prediction(table: pd.DataFrame) -> pd.DataFrame:
    """Rank drivers purely by (untransformed) qualifying time."""
    result = table.copy()
    raw_column = (
        "QualifyingTimeRaw (s)" if "QualifyingTimeRaw (s)" in result.columns else "QualifyingTime (s)"
    )
    result["PredictedRaceTime (s)"] = result[raw_column]
    return result.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
