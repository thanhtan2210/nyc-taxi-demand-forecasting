"""Downloads hourly observed weather for New York City from the Open-Meteo archive API.

Usage:
    python -m src.weather

The API is queried in GMT and converted to New York wall time here. Asking the API for
timezone=America/New_York applies one fixed UTC offset to the whole response, which shifts
every winter (EST) hour by one hour relative to the TLC pickup timestamps.
"""
import sys
from pathlib import Path

import pandas as pd
import requests

from .config import ROOT, load_config, month_range
from .timeutils import TIMEZONE, hours_of_months

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OUTPUT = ROOT / "data" / "external" / "weather_hourly.csv"
VARIABLES = ["temperature_2m", "precipitation"]


def fetch_weather(latitude, longitude, start_date, end_date, get=requests.get):
    """Returns the raw API payload for [start_date, end_date] (GMT days)."""
    response = get(
        API_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(VARIABLES),
            "timezone": "GMT",
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def to_local_hourly(payload, hours):
    """Converts a GMT payload to one row per New York wall-clock hour in `hours`.

    The hour repeated when clocks fall back keeps its first record. The hour skipped when
    clocks spring forward does not exist in the weather series and is forward-filled, so
    it only ever uses an earlier observation.
    """
    hourly = payload["hourly"]
    frame = pd.DataFrame({name: hourly[name] for name in VARIABLES})
    utc = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    frame.insert(0, "hour", utc.tz_convert(TIMEZONE).tz_localize(None))
    frame = frame.drop_duplicates("hour", keep="first").set_index("hour")
    frame = frame.reindex(pd.DatetimeIndex(hours, name="hour")).ffill()
    return frame.reset_index()


def main():
    cfg = load_config()
    hours = hours_of_months(month_range(cfg["months"]["start"], cfg["months"]["end"]))
    # One extra GMT day covers the New York evening hours of the last day.
    payload = fetch_weather(
        cfg["weather"]["latitude"],
        cfg["weather"]["longitude"],
        hours[0].strftime("%Y-%m-%d"),
        (hours[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    frame = to_local_hourly(payload, hours)
    missing = int(frame[VARIABLES].isna().any(axis=1).sum())
    if missing:
        print(f"{missing} hours have no weather value; refusing to write a partial file")
        return 1
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DEFAULT_OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(frame):,} hours to {Path(DEFAULT_OUTPUT).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
