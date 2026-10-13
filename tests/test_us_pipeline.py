import pandas as pd

from backend.ml.preprocess_us import load_bts
from backend.ml.weather import WEATHER_FEATURES


def test_us_feature_contract_is_explicit():
    assert WEATHER_FEATURES == ["TemperatureC", "PrecipitationMM", "WindSpeedKPH", "CloudCoverPct"]


def test_bts_loader_rejects_missing_required_schema(tmp_path):
    path = tmp_path / "not_bts.csv"
    pd.DataFrame({"FlightDate": ["2024-01-01"]}).to_csv(path, index=False)
    try:
        load_bts(path)
    except ValueError as error:
        assert "BTS file missing columns" in str(error)
    else:
        raise AssertionError("expected a schema error")
