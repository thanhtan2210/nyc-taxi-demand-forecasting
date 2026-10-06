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
# A day is unusual below 60% or above 140% of the median of the same weekday, 4 weeks either side.
UNUSUAL_LOW, UNUSUAL_HIGH, UNUSUAL_WEEKS = 0.6, 1.4, 4
# Below this mean volume a percentage error is too noisy to show.
MIN_TRIPS_FOR_WAPE = 5
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


def log10_trips(avg_trips_per_hour):
    """log10 of mean trips per hour for a map colour; zones below 1 trip per hour are drawn at 0."""
    return np.log10(np.maximum(np.asarray(avg_trips_per_hour, dtype=float), 1.0))


def power_of_ten_ticks(max_value):
    """Colourbar ticks for a log10 scale: ([0, 1, 2, ...], ["1", "10", "100", ...]) up to max_value."""
    top = max(1, int(np.ceil(np.log10(max(max_value, 1.0)))))
    exponents = list(range(top + 1))
    return exponents, [f"{10 ** e:,}" for e in exponents]


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


# --- Patterns ---------------------------------------------------------------------------

def borough_hourly(mart):
    """Trips per borough and hour: a small cube that the borough-level charts are built from."""
    return mart.groupby(["borough", "hour"], observed=True)[TRIP_COLUMNS].sum().reset_index()


def weekday_hour_heatmap(frame, column="trips_total"):
    """Mean trips per hour for each weekday (rows Mon..Sun) and hour of day (columns 0..23)."""
    hourly = frame.groupby("hour")[column].sum()
    cells = pd.DataFrame({
        "weekday": hourly.index.dayofweek, "hour_of_day": hourly.index.hour, "trips": hourly.to_numpy(),
    })
    table = cells.pivot_table(index="weekday", columns="hour_of_day", values="trips", aggfunc="mean")
    table = table.reindex(index=range(7), columns=range(24))
    table.index = WEEKDAYS
    return table


def monthly_trend(frame, by=None):
    """Mean trips per day in each month, for the whole frame or per value of the `by` column."""
    month = frame["hour"].dt.strftime("%Y-%m").rename("month")
    days = frame["hour"].dt.normalize().groupby(month).nunique().rename("n_days")
    keys = [month] if by is None else [month, frame[by]]
    totals = frame.groupby(keys, observed=True)["trips_total"].sum().reset_index()
    totals = totals.merge(days.reset_index(), on="month")
    totals["trips_per_day"] = totals["trips_total"] / totals["n_days"]
    return totals.drop(columns=["trips_total", "n_days"])


def month_over_month(trend, by):
    """Change (%) of trips per day against the previous month: months as rows, groups as columns."""
    wide = trend.pivot(index="month", columns=by, values="trips_per_day").sort_index()
    return 100 * wide.pct_change()


def service_mix_by_borough(frame):
    """Share (%) of each service within every borough (the shares of a borough add up to 100)."""
    trips = frame.groupby("borough", observed=True)[SERVICE_COLUMNS].sum()
    share = 100 * trips.div(trips.sum(axis=1).where(lambda s: s > 0), axis=0)
    share.columns = SERVICES
    trips.columns = SERVICES
    out = share.stack().rename("share_pct").reset_index().rename(columns={"level_1": "service"})
    out["trips"] = trips.stack().to_numpy()
    return out


def zone_kpis(zone_frame):
    """Total trips, mean trips per hour and the busiest hour of the day for one zone."""
    by_hour = zone_frame.groupby(zone_frame["hour"].dt.hour)["trips_total"].mean()
    return {
        "total_trips": int(zone_frame["trips_total"].sum()),
        "avg_trips_per_hour": float(zone_frame["trips_total"].mean()),
        "peak_hour_of_day": int(by_hour.idxmax()),
        "peak_hour_avg_trips": float(by_hour.max()),
    }


# --- Unusual days -----------------------------------------------------------------------

def unusual_days(daily, low=UNUSUAL_LOW, high=UNUSUAL_HIGH, weeks=UNUSUAL_WEEKS, min_reference_days=4):
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


# --- Forecast diagnostics ---------------------------------------------------------------

def hour_snapshot(backtest, hour):
    """Every zone at one hour: actual, baseline, XGBoost and the XGBoost error (forecast - actual)."""
    rows = backtest[backtest["hour"] == pd.Timestamp(hour)]
    out = rows[["zone_id", "actual", "baseline", "xgb"]].sort_values("zone_id").reset_index(drop=True)
    out["error"] = out["xgb"] - out["actual"]
    return out


def snapshot_kpis(snapshot):
    """Citywide totals and the MAE of both models over the zones of one hour."""
    return {
        "total_forecast": float(snapshot["xgb"].sum()),
        "total_actual": float(snapshot["actual"].sum()),
        "mae_xgboost": float((snapshot["xgb"] - snapshot["actual"]).abs().mean()),
        "mae_baseline": float((snapshot["baseline"] - snapshot["actual"]).abs().mean()),
    }


def mae_by_hour_of_day(rows):
    """MAE of the baseline and of XGBoost for each hour of the day, over the given zone-hours."""
    errors = pd.DataFrame({
        "hour_of_day": rows["hour"].dt.hour.to_numpy(),
        "mae_baseline": (rows["baseline"] - rows["actual"]).abs().to_numpy(),
        "mae_xgboost": (rows["xgb"] - rows["actual"]).abs().to_numpy(),
    })
    return errors.groupby("hour_of_day")[["mae_baseline", "mae_xgboost"]].mean().reset_index()


def zone_errors(rows, min_avg_trips=MIN_TRIPS_FOR_WAPE):
    """Per-zone error of both models over the given zone-hours.

    WAPE is a percentage of the zone's own volume, so it is only reported (`stable`) for zones
    that average at least `min_avg_trips` trips per hour; below that it is NaN.
    """
    frame = pd.DataFrame({
        "zone_id": rows["zone_id"].to_numpy(),
        "actual": rows["actual"].to_numpy(dtype=float),
        "abs_baseline": (rows["baseline"] - rows["actual"]).abs().to_numpy(dtype=float),
        "abs_xgboost": (rows["xgb"] - rows["actual"]).abs().to_numpy(dtype=float),
    })
    grouped = frame.groupby("zone_id")
    out = pd.DataFrame({
        "mean_actual": grouped["actual"].mean(),
        "mae_baseline": grouped["abs_baseline"].mean(),
        "mae_xgboost": grouped["abs_xgboost"].mean(),
    })
    out["stable"] = out["mean_actual"] >= min_avg_trips
    out["wape_xgboost"] = (100 * grouped["abs_xgboost"].sum() / grouped["actual"].sum()).where(out["stable"])
    return out.reset_index()


def largest_misses(rows, n=10):
    """The n zone-hours where XGBoost was furthest from the actual value."""
    out = rows[["hour", "zone_id", "actual", "baseline", "xgb"]].copy()
    out["error"] = out["xgb"] - out["actual"]
    order = out["error"].abs().sort_values(ascending=False, kind="stable").index[:n]
    return out.loc[order].reset_index(drop=True)
