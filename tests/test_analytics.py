"""Unit tests of the pure functions behind the app, on small hand-made frames."""
import numpy as np
import pandas as pd
import pytest

from src import analytics


def make_mart(start="2026-01-05", days=70, zones=((1, "Manhattan"), (2, "Queens"))):
    """A flat synthetic mart: every zone-hour has the same trips, so expected values are easy.

    Per zone and hour: fhvhv 12, fhv 2, green 2, yellow 4 (total 20). 2026-01-05 is a Monday.
    """
    hours = pd.date_range(start, periods=24 * days, freq="h")
    frames = []
    for zone_id, borough in zones:
        frames.append(pd.DataFrame({
            "hour": hours, "zone_id": zone_id, "borough": borough,
            "trips_fhvhv": 12, "trips_fhv": 2, "trips_green": 2, "trips_yellow": 4,
        }))
    mart = pd.concat(frames, ignore_index=True)
    mart["trips_total"] = mart[analytics.SERVICE_COLUMNS].sum(axis=1)
    return mart


@pytest.fixture
def mart():
    return make_mart()


QUALITY = [
    {"service": "yellow", "month": "2026-01", "rows_read": 100, "rows_outside_month": 2, "rows_zone_null": 0,
     "rows_zone_264_265": 8, "rows_zone_out_of_range": 0, "rows_in_mart": 90},
    {"service": "fhv", "month": "2026-01", "rows_read": 200, "rows_outside_month": 0, "rows_zone_null": 160,
     "rows_zone_264_265": 0, "rows_zone_out_of_range": 0, "rows_in_mart": 40},
    {"service": "fhv", "month": "2026-02", "rows_read": 100, "rows_outside_month": 0, "rows_zone_null": 90,
     "rows_zone_264_265": 0, "rows_zone_out_of_range": 0, "rows_in_mart": 10},
]


def test_overview_kpis_and_daily_totals(mart):
    kpis = analytics.overview_kpis(mart)
    assert kpis["total_trips"] == 2 * 70 * 24 * 20
    assert kpis["n_zones"] == 2
    assert kpis["first_hour"] == pd.Timestamp("2026-01-05 00:00")
    daily = analytics.daily_totals(mart)
    assert len(daily) == 70
    assert (daily["trips_total"] == 2 * 24 * 20).all()
    assert (daily["trips_fhvhv"] == 2 * 24 * 12).all()


def test_service_share_sums_to_100_in_fixed_order(mart):
    share = analytics.service_share(mart)
    assert share["service"].tolist() == ["fhvhv", "fhv", "green", "yellow"]
    assert share["share_pct"].tolist() == pytest.approx([60, 10, 10, 20])
    assert share["trips"].sum() == mart["trips_total"].sum()
    # also works on a subset, e.g. one zone
    assert analytics.service_share(mart[mart["zone_id"] == 1])["share_pct"].sum() == pytest.approx(100)


def test_hourly_profile_separates_weekdays_and_weekends(mart):
    mart = mart.copy()
    weekend_noon = (mart["hour"].dt.dayofweek >= 5) & (mart["hour"].dt.hour == 12)
    mart.loc[weekend_noon, "trips_total"] = 50
    profile = analytics.hourly_profile(mart).set_index(["day_type", "hour_of_day"])["trips_per_hour"]
    assert len(profile) == 48
    assert profile["Weekday", 12] == 40   # two zones x 20
    assert profile["Weekend", 12] == 100  # two zones x 50
    assert profile["Weekend", 13] == 40


def test_zone_totals_and_top_zones(mart):
    mart = mart.copy()
    mart.loc[mart["zone_id"] == 2, "trips_total"] = 30
    zones = pd.DataFrame({"zone_id": [1, 2, 264], "zone": ["A", "B", "N/A"], "borough": ["Manhattan", "Queens", "Unknown"]})
    table = analytics.zone_totals(mart, zones).set_index("zone_id")
    assert table.loc[2, "avg_trips_per_hour"] == 30
    assert table.loc[1, "trips_total"] == 70 * 24 * 20
    assert table.loc[2, "zone"] == "B"
    assert analytics.top_zones(table.reset_index(), 1)["zone_id"].tolist() == [2]


def test_source_vs_mart_adds_up_and_finds_the_largest_gap():
    table = analytics.source_vs_mart(QUALITY).set_index("service")
    assert table.index.tolist() == ["yellow", "fhv", "all"]
    assert table.loc["fhv", "rows_read"] == 300
    assert table.loc["fhv", "rows_zone_null"] == 250
    assert table.loc["all", "rows_read"] == 400
    assert table.loc["all", "rows_in_mart"] == 140
    assert table.loc["all", "pct_in_mart"] == pytest.approx(35.0)
    assert analytics.largest_gap(table.reset_index()) == ("fhv", "rows_zone_null", 250)


def test_fhv_null_zone_share_per_month():
    share = analytics.fhv_null_zone_share(QUALITY)
    assert share.to_dict() == pytest.approx({"2026-01": 80.0, "2026-02": 90.0})


def test_unusual_days_finds_a_deliberate_drop_and_its_services(mart):
    drop_day = pd.Timestamp("2026-02-11")  # a Wednesday with four weeks of data on both sides
    on_day = mart["hour"].dt.normalize() == drop_day
    # every service falls to half, except yellow which falls to zero -> total = 4/10 of normal
    mart.loc[on_day, ["trips_fhvhv", "trips_fhv", "trips_green", "trips_yellow"]] = [6, 1, 1, 0]
    mart["trips_total"] = mart[analytics.SERVICE_COLUMNS].sum(axis=1)

    found = analytics.unusual_days(analytics.daily_totals(mart))
    assert found["date"].tolist() == [drop_day]
    row = found.iloc[0]
    assert row["weekday"] == "Wed"
    assert row["pct_of_normal"] == pytest.approx(40.0)
    assert row["pct_fhvhv"] == pytest.approx(50.0)
    assert row["pct_yellow"] == pytest.approx(0.0)
    assert row["trips_total"] == 2 * 24 * 8


def test_unusual_days_thresholds_and_reference_exclude_the_day_itself(mart):
    daily = analytics.daily_totals(mart).astype(float)
    assert analytics.unusual_days(daily).empty  # a flat series has no unusual day

    spike = daily.copy()
    spike.loc["2026-02-11"] *= 1.5
    found = analytics.unusual_days(spike)
    assert found["date"].tolist() == [pd.Timestamp("2026-02-11")]
    assert found.iloc[0]["pct_of_normal"] == pytest.approx(150.0)  # the spike is not part of its own reference
    assert analytics.unusual_days(spike, high=1.6).empty

    # 59% is flagged, 61% is not
    for factor, expected in [(0.59, 1), (0.61, 0)]:
        dip = daily.copy()
        dip.loc["2026-02-11"] *= factor
        assert len(analytics.unusual_days(dip)) == expected


def test_unusual_days_ignores_days_without_enough_reference_days():
    short = analytics.daily_totals(make_mart(days=21)).astype(float)  # at most 2 same-weekday neighbours
    short.iloc[10] *= 0.1
    assert analytics.unusual_days(short).empty
    relaxed = analytics.unusual_days(short, min_reference_days=2).set_index("date")
    assert relaxed.loc[pd.Timestamp("2026-01-15"), "pct_of_normal"] == pytest.approx(10.0)


def test_month_weeks_cover_the_month_without_overlap():
    weeks = analytics.month_weeks("2026-05")
    assert weeks[0] == (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-08"))
    assert weeks[-1] == (pd.Timestamp("2026-05-29"), pd.Timestamp("2026-06-01"))
    assert len(weeks) == 5
    assert all(a[1] == b[0] for a, b in zip(weeks, weeks[1:]))
    assert len(analytics.month_weeks("2026-02")) == 4


def test_backtest_series_sums_zones_per_hour():
    hours = pd.date_range("2026-05-01", periods=48, freq="h")
    backtest = pd.concat([
        pd.DataFrame({"hour": hours, "zone_id": z, "actual": 10.0, "baseline": 8.0, "xgb": 9.0}) for z in (1, 2)
    ])
    out = analytics.backtest_series(backtest, pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-02"))
    assert len(out) == 24
    assert (out["actual"] == 20).all() and (out["baseline"] == 16).all() and (out["xgb"] == 18).all()
    assert out["hour"].max() == pd.Timestamp("2026-05-01 23:00")
    assert not np.isnan(out[["actual", "baseline", "xgb"]].to_numpy()).any()


def test_weekday_hour_heatmap_shape_and_values(mart):
    mart = mart.copy()
    friday_evening = (mart["hour"].dt.dayofweek == 4) & (mart["hour"].dt.hour == 18)
    mart.loc[friday_evening, "trips_total"] = 100
    table = analytics.weekday_hour_heatmap(mart)
    assert table.shape == (7, 24)
    assert table.index.tolist() == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert table.loc["Fri", 18] == 200  # two zones x 100
    assert table.loc["Fri", 17] == 40
    assert table.loc["Mon", 18] == 40
    # one service, one borough
    queens = analytics.borough_hourly(mart)
    queens = queens[queens["borough"] == "Queens"]
    assert (analytics.weekday_hour_heatmap(queens, "trips_yellow").to_numpy() == 4).all()


def test_borough_hourly_keeps_totals(mart):
    cube = analytics.borough_hourly(mart)
    assert len(cube) == 2 * 70 * 24
    assert cube["trips_total"].sum() == mart["trips_total"].sum()
    assert set(cube["borough"]) == {"Manhattan", "Queens"}


def test_monthly_trend_and_month_over_month():
    mart = make_mart(start="2026-01-01", days=90)  # January, February and March 2026
    in_feb = mart["hour"].dt.month == 2
    mart.loc[in_feb & (mart["zone_id"] == 1), "trips_total"] = 30  # Manhattan +50% in February
    trend = analytics.monthly_trend(mart, by="borough")
    wide = trend.pivot(index="month", columns="borough", values="trips_per_day")
    assert wide.index.tolist() == ["2026-01", "2026-02", "2026-03"]
    assert wide.loc["2026-01", "Manhattan"] == 24 * 20
    assert wide.loc["2026-02", "Manhattan"] == 24 * 30  # per day, so month length does not matter
    assert wide.loc["2026-02", "Queens"] == 24 * 20

    change = analytics.month_over_month(trend, by="borough")
    assert np.isnan(change.loc["2026-01", "Manhattan"])
    assert change.loc["2026-02", "Manhattan"] == pytest.approx(50.0)
    assert change.loc["2026-03", "Manhattan"] == pytest.approx(-100 / 3)
    assert change.loc["2026-02", "Queens"] == pytest.approx(0.0)

    # without a grouping column: one row per month for the whole frame
    city = analytics.monthly_trend(mart)
    assert city.columns.tolist() == ["month", "trips_per_day"]
    assert city.set_index("month").loc["2026-01", "trips_per_day"] == 2 * 24 * 20


def test_service_mix_by_borough_rows_add_up_to_100(mart):
    mart = mart.copy()
    mart.loc[mart["borough"] == "Queens", "trips_yellow"] = 0
    mix = analytics.service_mix_by_borough(mart)
    assert mix.groupby("borough", observed=True)["share_pct"].sum().tolist() == pytest.approx([100, 100])
    wide = mix.pivot(index="borough", columns="service", values="share_pct")
    assert wide.loc["Manhattan"].to_dict() == pytest.approx({"fhvhv": 60, "fhv": 10, "green": 10, "yellow": 20})
    assert wide.loc["Queens", "yellow"] == 0
    assert wide.loc["Queens", "fhvhv"] == pytest.approx(75.0)
    assert mix["trips"].sum() == mart[analytics.SERVICE_COLUMNS].to_numpy().sum()


def test_zone_kpis(mart):
    zone = mart[mart["zone_id"] == 1].copy()
    zone.loc[zone["hour"].dt.hour == 8, "trips_total"] = 44
    kpis = analytics.zone_kpis(zone)
    assert kpis["peak_hour_of_day"] == 8
    assert kpis["peak_hour_avg_trips"] == 44
    assert kpis["total_trips"] == 70 * (23 * 20 + 44)
    assert kpis["avg_trips_per_hour"] == pytest.approx((23 * 20 + 44) / 24)


def make_backtest():
    """Two zones over two days. Zone 1 is busy, zone 2 averages 2 trips per hour."""
    hours = pd.date_range("2026-05-01", periods=48, freq="h")
    busy = pd.DataFrame({"hour": hours, "zone_id": 1, "actual": 100.0, "baseline": 90.0, "xgb": 104.0})
    quiet = pd.DataFrame({"hour": hours, "zone_id": 2, "actual": 2.0, "baseline": 4.0, "xgb": 3.0})
    return pd.concat([busy, quiet], ignore_index=True)


def test_mae_by_hour_of_day():
    backtest = make_backtest()
    backtest.loc[(backtest["zone_id"] == 1) & (backtest["hour"].dt.hour == 9), "xgb"] = 130.0
    table = analytics.mae_by_hour_of_day(backtest).set_index("hour_of_day")
    assert len(table) == 24
    assert table.loc[0, "mae_baseline"] == pytest.approx((10 + 2) / 2)
    assert table.loc[0, "mae_xgboost"] == pytest.approx((4 + 1) / 2)
    assert table.loc[9, "mae_xgboost"] == pytest.approx((30 + 1) / 2)
    # one zone only
    busy = analytics.mae_by_hour_of_day(backtest[backtest["zone_id"] == 1]).set_index("hour_of_day")
    assert busy.loc[0, "mae_xgboost"] == pytest.approx(4)


def test_zone_errors_hides_wape_below_five_trips_per_hour():
    table = analytics.zone_errors(make_backtest()).set_index("zone_id")
    assert table.loc[1, "mean_actual"] == 100
    assert table.loc[1, "mae_xgboost"] == pytest.approx(4) and table.loc[1, "mae_baseline"] == pytest.approx(10)
    assert table.loc[1, "wape_xgboost"] == pytest.approx(4.0)
    assert bool(table.loc[1, "stable"])
    # zone 2 averages 2 trips per hour: MAE is kept, the percentage is not
    assert not bool(table.loc[2, "stable"])
    assert np.isnan(table.loc[2, "wape_xgboost"])
    assert table.loc[2, "mae_xgboost"] == pytest.approx(1)
    # the threshold is inclusive and configurable
    assert analytics.zone_errors(make_backtest(), min_avg_trips=2).set_index("zone_id").loc[2, "wape_xgboost"] == pytest.approx(50.0)
    assert analytics.MIN_TRIPS_FOR_WAPE == 5


def test_largest_misses_are_sorted_by_absolute_error():
    backtest = make_backtest()
    backtest.loc[5, "xgb"] = 400.0   # +300 over actual
    backtest.loc[60, "xgb"] = 2.0    # exact, never listed first
    backtest.loc[20, "xgb"] = 0.0    # -100 under actual
    top = analytics.largest_misses(backtest, n=3)
    assert top["error"].tolist() == [300.0, -100.0, 4.0]
    assert top.loc[0, "hour"] == pd.Timestamp("2026-05-01 05:00") and top.loc[0, "zone_id"] == 1
    assert top.columns.tolist() == ["hour", "zone_id", "actual", "baseline", "xgb", "error"]
    assert len(analytics.largest_misses(backtest)) == 10


def test_log10_trips_floors_at_one_trip():
    assert analytics.log10_trips([0, 0.5, 1, 10, 250]).tolist() == pytest.approx([0, 0, 0, 1, np.log10(250)])


def test_log_scale_ends_at_the_largest_value_not_at_the_next_power_of_ten():
    upper, values, labels = analytics.log_scale_ticks(1712.4)
    assert upper == pytest.approx(np.log10(1712.4))
    assert values == pytest.approx([0, 1, 2, 3, np.log10(1712.4)])
    assert labels == ["1", "10", "100", "1,000", "1,712"]

    upper, values, labels = analytics.log_scale_ticks(706.4)
    assert upper == pytest.approx(np.log10(706.4)) and upper < 3
    assert labels == ["1", "10", "100", "706"]

    # only powers of ten strictly below the maximum get a tick
    assert analytics.log_scale_ticks(100)[2] == ["1", "10", "100"]
    assert analytics.log_scale_ticks(100)[1] == pytest.approx([0, 1, 2])
    # a power of ten that would sit on top of the maximum label is dropped
    assert analytics.log_scale_ticks(1040)[2] == ["1", "10", "100", "1,040"]
    assert analytics.log_scale_ticks(0.3) == (0.0, [0.0], ["1"])


def test_hour_snapshot_returns_every_zone_of_that_hour_only():
    backtest = make_backtest()
    moment = pd.Timestamp("2026-05-02 18:00")
    backtest.loc[(backtest["hour"] == moment) & (backtest["zone_id"] == 1), ["actual", "xgb"]] = [120.0, 111.5]
    snapshot = analytics.hour_snapshot(backtest, moment)
    assert snapshot["zone_id"].tolist() == [1, 2]
    assert snapshot.columns.tolist() == ["zone_id", "actual", "baseline", "xgb", "error"]
    assert snapshot.loc[0, ["actual", "baseline", "xgb", "error"]].tolist() == [120.0, 90.0, 111.5, -8.5]
    assert snapshot.loc[1, "error"] == 1.0
    # the neighbouring hours are untouched and a string timestamp works too
    assert analytics.hour_snapshot(backtest, "2026-05-02 17:00").loc[0, "actual"] == 100.0
    assert analytics.hour_snapshot(backtest, "2027-01-01 00:00").empty


def test_snapshot_kpis_are_citywide():
    snapshot = analytics.hour_snapshot(make_backtest(), "2026-05-01 18:00")
    kpis = analytics.snapshot_kpis(snapshot)
    assert kpis["total_forecast"] == 104 + 3
    assert kpis["total_actual"] == 100 + 2
    assert kpis["mae_xgboost"] == pytest.approx((4 + 1) / 2)
    assert kpis["mae_baseline"] == pytest.approx((10 + 2) / 2)
