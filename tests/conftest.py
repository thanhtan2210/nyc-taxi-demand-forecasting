import pytest
import polars as pl
import pandas as pd
from datetime import datetime

@pytest.fixture
def sample_yellow_raw():
    """Generates a sample Polars LazyFrame simulating yellow taxi raw data."""
    data = {
        "tpep_pickup_datetime": [
            datetime(2025, 6, 15, 12, 0, 0),  # Valid
            datetime(2025, 6, 15, 13, 0, 0),  # Valid
            datetime(2025, 5, 29, 12, 0, 0),  # Out of date bounds (May)
            datetime(2025, 6, 15, 12, 0, 0),  # Outlier distance (0.0)
            datetime(2025, 6, 15, 12, 0, 0),  # Outlier fare (1.0)
            datetime(2025, 6, 15, 12, 0, 0),  # Null location ids
            datetime(2025, 6, 15, 12, 0, 0)   # Invalid duration (dropoff before pickup or same)
        ],
        "tpep_dropoff_datetime": [
            datetime(2025, 6, 15, 12, 30, 0), # 30 min duration
            datetime(2025, 6, 15, 13, 15, 0), # 15 min duration
            datetime(2025, 5, 29, 12, 30, 0),
            datetime(2025, 6, 15, 12, 10, 0),
            datetime(2025, 6, 15, 12, 10, 0),
            datetime(2025, 6, 15, 12, 10, 0),
            datetime(2025, 6, 15, 12, 0, 0)   # 0 min duration
        ],
        "pulocationid": [1, 2, 3, 4, 5, None, 7],
        "dolocationid": [10, 20, 30, 40, 50, 60, None],
        "passenger_count": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "trip_distance": [2.5, 5.0, 1.5, 0.0, 3.0, 2.0, 1.0], # 0.0 is outlier
        "fare_amount": [12.5, 20.0, 8.0, 10.0, 1.0, 15.0, 10.0] # 1.0 is outlier
    }
    return pl.LazyFrame(data)

@pytest.fixture
def sample_fhvhv_raw():
    """Generates a sample Polars LazyFrame simulating FHVHV raw data."""
    data = {
        "pickup_datetime": [
            datetime(2025, 7, 10, 8, 0, 0),   # Valid
            datetime(2025, 7, 10, 9, 0, 0),   # Valid
            datetime(2025, 7, 10, 8, 0, 0)    # Distance outlier (>100 miles)
        ],
        "dropoff_datetime": [
            datetime(2025, 7, 10, 8, 45, 0),
            datetime(2025, 7, 10, 9, 15, 0),
            datetime(2025, 7, 10, 11, 0, 0)
        ],
        "PULocationID": [100, 101, 102],
        "DOLocationID": [200, 201, 202],
        "base_passenger_fare": [25.0, 30.0, 150.0],
        "trip_miles": [8.5, 12.0, 150.0], # 150.0 is outlier
        "shared_request_flag": ["Y", None, "N"],
        "shared_match_flag": ["N", "N", None]
    }
    return pl.LazyFrame(data)

@pytest.fixture
def sample_bq_features_data():
    """Generates a Pandas DataFrame simulating the output of fetch_data_from_bq."""
    data = {
        "Time_Key": [
            2025060100, 2025060101, 2025060102, 2025060103,
            2025060100, 2025060101, 2025060102, 2025060103
        ],
        "PULocation_Key": [1, 1, 1, 1, 2, 2, 2, 2],
        "total_demand": [10, 15, 20, 25, 5, 8, 4, 12],
        "Temperature": [72.5, 71.0, 70.0, 69.5, 72.5, 71.0, 70.0, 69.5],
        "Precipitation": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        "DayOfWeek": [6, 6, 6, 6, 6, 6, 6, 6] # Sunday
    }
    return pd.DataFrame(data)
