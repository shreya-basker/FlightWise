import pandas as pd
import pytest

from backend.ml.preprocess import preprocess_dataframe


def test_preprocessing_keeps_only_predeparture_features_and_target():
    raw = pd.DataFrame({
        "FlightDate": ["2025-01-01", "2025-01-02"], "Airline": [" Example Air ", "Example Air"],
        "Origin": ["DEL", "DEL"], "Dest": ["BOM", "BOM"], "CRSDepTime": ["08:30", "2460"],
        "DepDelay": [20, 0], "Cancelled": [0, 0], "Distance_KM": [1148, 1148], "ArrDelay": [99, 0],
    })
    cleaned = preprocess_dataframe(raw)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "Delayed"] == 1
    assert "ArrDelay" not in cleaned.columns
    assert cleaned.loc[0, "DayOfWeek"] == 2


def test_preprocessing_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        preprocess_dataframe(pd.DataFrame({"FlightDate": ["2025-01-01"]}))

