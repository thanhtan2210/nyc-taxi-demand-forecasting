import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.build_features import engineer_features

@pytest.fixture
def long_period_features_data():
    """Generates 180 consecutive hours of demand data to test lag features (lag_168h)."""
    start_time = datetime(2025, 6, 1, 0, 0, 0)
    times = [start_time + timedelta(hours=i) for i in range(180)]
    
    data = {
        "Time_Key": [int(t.strftime("%Y%m%d%H")) for t in times],
        "PULocation_Key": [1] * 180,
        "total_demand": [float(10 + (i % 10)) for i in range(180)],
        "Temperature": [72.0 + (i % 5) for i in range(180)],
        "Precipitation": [None if i % 10 == 0 else 0.0 for i in range(180)] # Include some nulls
    }
    # Add a single null temperature to test imputation
    data["Temperature"][5] = None
    
    return pd.DataFrame(data)

def test_engineer_features_creates_correct_columns(long_period_features_data):
    df_out = engineer_features(long_period_features_data)
    
    # 1. Check expected output columns exist
    expected_cols = {
        "Datetime", "Hour", "DayOfWeek", "Is_Weekend",
        "hour_sin", "hour_cos", "lag_1h", "lag_2h", "lag_24h", "lag_168h",
        "rolling_mean_6h", "Temperature", "Precipitation"
    }
    assert expected_cols.issubset(set(df_out.columns))

def test_engineer_features_cyclical_encoding(long_period_features_data):
    df_out = engineer_features(long_period_features_data)
    
    # Check cyclical mathematical properties: sin^2 + cos^2 = 1
    assert np.allclose(df_out["hour_sin"]**2 + df_out["hour_cos"]**2, 1.0)

def test_engineer_features_lags_and_dropna(long_period_features_data):
    df_out = engineer_features(long_period_features_data)
    
    # Since the input has 180 hours and we drop rows where lag_168h is null:
    # 180 - 168 = 12 rows should remain.
    assert len(df_out) == 12
    assert df_out["lag_168h"].isnull().sum() == 0
    
    # Verify exact lag value: lag_1h of row index 0 (which corresponds to index 168 in original data)
    # should be the demand at index 167 in original data.
    # original index 168 total_demand = 10 + (168 % 10) = 18.
    # original index 167 total_demand = 10 + (167 % 10) = 17.
    # lag_1h at output index 0 should be 17.
    assert df_out.iloc[0]["lag_1h"] == 17.0
    assert df_out.iloc[0]["lag_168h"] == 10.0 # Index 0 total_demand = 10 + (0 % 10) = 10.

def test_engineer_features_imputation(long_period_features_data):
    df_out = engineer_features(long_period_features_data)
    
    # Null temperature and precipitation should be imputed
    assert df_out["Temperature"].isnull().sum() == 0
    assert df_out["Precipitation"].isnull().sum() == 0
