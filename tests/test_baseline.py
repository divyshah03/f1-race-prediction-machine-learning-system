import pandas as pd

from f1_predictor.models.baseline import baseline_prediction


def test_baseline_prediction_ranks_by_qualifying_time():
    table = pd.DataFrame(
        {
            "Driver": ["A", "B", "C"],
            "QualifyingTime (s)": [82.0, 80.0, 81.0],
        }
    )

    result = baseline_prediction(table)

    assert result["Driver"].tolist() == ["B", "C", "A"]
    assert result["PredictedRaceTime (s)"].tolist() == [80.0, 81.0, 82.0]
