"""Weather tests with a mocked Open-Meteo response (no network)."""
import pandas as pd

from src.weather import API_URL, fetch_weather, to_local_hourly


def gmt_payload(start, end):
    """A fake API payload whose temperature equals the GMT hour index, so rows are traceable."""
    times = pd.date_range(start, end, freq="h")
    return {
        "utc_offset_seconds": 0,
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [float(i) for i in range(len(times))],
            "precipitation": [0.1] * len(times),
        },
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_fetch_requests_gmt_hourly_variables():
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return FakeResponse(gmt_payload("2026-06-01 00:00", "2026-06-01 23:00"))

    payload = fetch_weather(40.7128, -74.0060, "2026-06-01", "2026-06-01", get=fake_get)
    url, params = calls[0]
    assert url == API_URL
    assert params["timezone"] == "GMT"
    assert params["hourly"] == "temperature_2m,precipitation"
    assert (params["latitude"], params["longitude"]) == (40.7128, -74.0060)
    assert len(payload["hourly"]["time"]) == 24


def test_summer_and_winter_hours_use_their_own_utc_offset():
    hours = pd.date_range("2026-01-15 00:00", "2026-01-15 03:00", freq="h")
    winter = to_local_hourly(gmt_payload("2026-01-15 00:00", "2026-01-16 23:00"), hours)
    assert winter["temperature_2m"].tolist() == [5.0, 6.0, 7.0, 8.0]  # EST = GMT-5

    hours = pd.date_range("2026-07-15 00:00", "2026-07-15 03:00", freq="h")
    summer = to_local_hourly(gmt_payload("2026-07-15 00:00", "2026-07-16 23:00"), hours)
    assert summer["temperature_2m"].tolist() == [4.0, 5.0, 6.0, 7.0]  # EDT = GMT-4


def test_fall_back_hour_keeps_the_first_record():
    hours = pd.date_range("2025-11-02 00:00", "2025-11-02 03:00", freq="h")
    out = to_local_hourly(gmt_payload("2025-11-02 00:00", "2025-11-03 23:00"), hours)
    # 01:00 happens twice (GMT 05:00 in EDT and GMT 06:00 in EST); the first one is kept
    assert out["temperature_2m"].tolist() == [4.0, 5.0, 7.0, 8.0]
    assert not out["hour"].duplicated().any()


def test_spring_forward_hour_is_filled_from_the_previous_hour():
    hours = pd.date_range("2026-03-08 00:00", "2026-03-08 04:00", freq="h")
    out = to_local_hourly(gmt_payload("2026-03-08 00:00", "2026-03-09 23:00"), hours)
    # 02:00 does not exist on the clock; it reuses the 01:00 observation
    assert out["temperature_2m"].tolist() == [5.0, 6.0, 6.0, 7.0, 8.0]
    assert not out.isna().any().any()
