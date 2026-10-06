"""Calendar attributes of local New York hours, shared by the warehouse and the features."""
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TIMEZONE = "America/New_York"


def hour_calendar(hours):
    """Returns one row per hour with date, hour of day, weekday and the two calendar flags.

    `hours` are naive timestamps in New York wall time (the convention of the TLC files).
    """
    hours = pd.DatetimeIndex(hours)
    holidays = USFederalHolidayCalendar().holidays(hours.min().normalize(), hours.max().normalize())
    # Wall-clock hours that are repeated (fall back) or skipped (spring forward) do not localize.
    localized = hours.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return pd.DataFrame({
        "hour": hours,
        "date": hours.date,
        "hour_of_day": hours.hour.astype("int8"),
        "day_of_week": hours.dayofweek.astype("int8"),  # Monday = 0
        "is_holiday": hours.normalize().isin(holidays),
        "is_dst_transition": localized.isna(),
    })


def month_bounds(month):
    """Returns [start, end) timestamps of a 'YYYY-MM' month."""
    start = pd.Timestamp(month + "-01")
    return start, start + pd.offsets.MonthBegin(1)


def hours_of_months(months):
    """Every hour from the first hour of the first month to the last hour of the last month."""
    start, _ = month_bounds(months[0])
    _, end = month_bounds(months[-1])
    return pd.date_range(start, end, freq="h", inclusive="left")
