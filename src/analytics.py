"""Pure aggregation functions behind the Streamlit pages.

Every function takes plain DataFrames and returns plain data, so it can be unit tested
without Streamlit. The pages only draw what these functions return.

`mart` is the hourly demand mart: one row per (hour, zone_id) with `borough`, `trips_total`
and one `trips_<service>` column per service. Functions that take a `frame` accept any subset
of mart rows (one zone, one borough, the whole city).
"""
import numpy as np
import pandas as pd

SERVICES = ["fhvhv", "fhv", "green", "yellow"]
SERVICE_COLUMNS = [f"trips_{s}" for s in SERVICES]
TRIP_COLUMNS = ["trips_total"] + SERVICE_COLUMNS
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DROP_REASONS = {
    "rows_outside_month": "with a pickup time outside the month",
    "rows_zone_null": "without a pickup zone",
    "rows_zone_264_265": "in zone 264/265 (unknown or outside NYC)",
    "rows_zone_out_of_range": "with an invalid zone id",
}


# --- Overview ---------------------------------------------------------------------------

def overview_kpis(mart):
    return {
        "total_trips": int(mart["trips_total"].sum()),
        "first_hour": mart["hour"].min(),
        "last_hour": mart["hour"].max(),
        "n_zones": int(mart["zone_id"].nunique()),
    }


def daily_totals(frame):
    """Trips per calendar day: total and per service, indexed by date (every day present)."""
    daily = frame.groupby(frame["hour"].dt.normalize())[TRIP_COLUMNS].sum()
    daily.index.name = "date"
    return daily.asfreq("D", fill_value=0)


def service_share(frame):
    """Trips and share (%) of each service, in the fixed service order."""
    trips = frame[SERVICE_COLUMNS].sum().to_numpy(dtype="int64")
    total = trips.sum()
    share = 100 * trips / total if total else np.zeros(len(trips))
    return pd.DataFrame({"service": SERVICES, "trips": trips, "share_pct": share})


def hourly_profile(frame):
    """Mean trips per hour of the day, for weekdays and for weekend days."""
    hourly = frame.groupby("hour")["trips_total"].sum().reset_index()
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_type"] = np.where(hourly["hour"].dt.dayofweek >= 5, "Weekend", "Weekday")
    return (hourly.groupby(["day_type", "hour_of_day"])["trips_total"].mean()
            .rename("trips_per_hour").reset_index())


def zone_totals(mart, zones):
    """Total trips and mean trips per hour of every zone in the mart, with its names."""
    n_hours = mart["hour"].nunique()
    totals = mart.groupby("zone_id")["trips_total"].sum().reset_index()
    totals["avg_trips_per_hour"] = totals["trips_total"] / n_hours
    return totals.merge(zones[["zone_id", "zone", "borough"]], on="zone_id", how="left")


def top_zones(zone_table, n=15):
    return zone_table.nlargest(n, "trips_total").reset_index(drop=True)


def source_vs_mart(entries):
    """Row accounting per service over all months, plus an 'all' row, from data_quality.json."""
    counts = ["rows_read", *DROP_REASONS, "rows_in_mart"]
    table = pd.DataFrame(entries).groupby("service", sort=False)[counts].sum()
    table = table.reindex([s for s in ["yellow", "green", "fhv", "fhvhv"] if s in table.index])
    table.loc["all"] = table.sum()
    table["pct_in_mart"] = 100 * table["rows_in_mart"] / table["rows_read"]
    return table.reset_index()


def largest_gap(source_table):
    """The (service, reason, rows) that removes the most source rows."""
    services = source_table[source_table["service"] != "all"].set_index("service")
    dropped = services[list(DROP_REASONS)].stack()
    service, reason = dropped.idxmax()
    return service, reason, int(dropped.max())


def fhv_null_zone_share(entries):
    """Share (%) of FHV rows without a pickup zone, per month."""
    fhv = pd.DataFrame([e for e in entries if e["service"] == "fhv"])
    return (100 * fhv["rows_zone_null"] / fhv["rows_read"]).set_axis(fhv["month"]).rename("pct_zone_null")


# --- Unusual days -----------------------------------------------------------------------

def unusual_days(daily, low=0.6, high=1.4, weeks=4, min_reference_days=4):
    """Days whose total is below `low` or above `high` times their normal level.

    The normal level of a day is the median of the same weekday in the `weeks` weeks before
    and the `weeks` weeks after it (the day itself is excluded). The same reference is
    computed for every service column so a drop can be traced to the services behind it.
    Days with fewer than `min_reference_days` reference days are not judged.
    """
    values = daily[TRIP_COLUMNS].astype(float)
    offsets = [k for k in range(-weeks, weeks + 1) if k != 0]
    neighbours = np.stack([values.shift(7 * k).to_numpy() for k in offsets])
    enough = (~np.isnan(neighbours[:, :, 0])).sum(axis=0) >= min_reference_days
    with np.errstate(all="ignore"):
        reference = pd.DataFrame(np.nanmedian(neighbours, axis=0), index=values.index, columns=TRIP_COLUMNS)
    ratio = values / reference.where(reference > 0)

    total_ratio = ratio["trips_total"]
    flagged = enough & ((total_ratio < low) | (total_ratio > high)).to_numpy()
    out = pd.DataFrame({
        "date": values.index[flagged],
        "weekday": [WEEKDAYS[d] for d in values.index[flagged].dayofweek],
        "trips_total": values.loc[flagged, "trips_total"].astype("int64").to_numpy(),
        "pct_of_normal": 100 * total_ratio[flagged].to_numpy(),
    })
    for service, column in zip(SERVICES, SERVICE_COLUMNS):
        out[f"pct_{service}"] = 100 * ratio.loc[flagged, column].to_numpy()
    return out.reset_index(drop=True)


# --- Forecast ---------------------------------------------------------------------------

def month_weeks(month):
    """Seven-day windows of a 'YYYY-MM' month, starting on its first day: [(start, end), ...].

    `end` is exclusive; the last window is shorter when the month does not divide by seven.
    """
    start = pd.Timestamp(month + "-01")
    month_end = start + pd.offsets.MonthBegin(1)
    starts = pd.date_range(start, month_end, freq="7D", inclusive="left")
    return [(s, min(s + pd.Timedelta(days=7), month_end)) for s in starts]


def backtest_series(backtest, start, end):
    """Hourly actual, baseline and XGBoost between start and end, summed over the given rows."""
    window = backtest[(backtest["hour"] >= start) & (backtest["hour"] < end)]
    return window.groupby("hour")[["actual", "baseline", "xgb"]].sum().reset_index()
