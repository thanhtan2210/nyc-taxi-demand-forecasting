import pytest
import pandas as pd

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
