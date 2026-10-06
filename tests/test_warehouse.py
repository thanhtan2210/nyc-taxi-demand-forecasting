"""Warehouse tests on small synthetic parquet files (no network)."""
import json
from datetime import datetime

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import load_config
from src.timeutils import hour_calendar, hours_of_months
from src.warehouse import run

MONTH = "2026-06"


def write_parquet(path, columns):
    pq.write_table(pa.table(columns), path)


@pytest.fixture
def source(tmp_path):
    """A fake TLC bucket holding one month of the four services plus the zone lookup."""
    src = tmp_path / "source"
    src.mkdir()
    # yellow: carries the extra request_source column that TLC added in 2026-06
    write_parquet(src / f"yellow_tripdata_{MONTH}.parquet", {
        "VendorID": pa.array([1] * 9, pa.int32()),
        "tpep_pickup_datetime": pa.array([
            datetime(2026, 6, 1, 0, 10),    # zone 1, hour 00
            datetime(2026, 6, 1, 0, 50),    # zone 1, hour 00
            datetime(2026, 6, 1, 2, 5),     # zone 2, hour 02
            datetime(2008, 12, 31, 23, 3),  # corrupt year -> outside month
            datetime(2026, 7, 1, 0, 0),     # first instant of next month -> outside month
            datetime(2026, 6, 30, 23, 59),  # last hour of the month, zone 1
            datetime(2026, 6, 2, 8, 0),     # zone null
            datetime(2026, 6, 2, 8, 0),     # zone 265
            datetime(2026, 6, 2, 9, 0),     # zone 264
        ], pa.timestamp("us")),
        "PULocationID": pa.array([1, 1, 2, 1, 1, 1, None, 265, 264], pa.int32()),
        "request_source": pa.array([None, "HV0003", None, None, None, "A", None, None, None], pa.large_string()),
    })
    write_parquet(src / f"green_tripdata_{MONTH}.parquet", {
        "lpep_pickup_datetime": pa.array([datetime(2026, 6, 1, 0, 30)], pa.timestamp("us")),
        "PULocationID": pa.array([1], pa.int32()),
    })
    # fhv: zone column is a float named PUlocationID and is mostly null
    write_parquet(src / f"fhv_tripdata_{MONTH}.parquet", {
        "pickup_datetime": pa.array([datetime(2026, 6, 1, 0, 0), datetime(2026, 6, 1, 0, 1),
                                     datetime(2026, 6, 1, 0, 2)], pa.timestamp("us")),
        "PUlocationID": pa.array([None, None, 2.0], pa.float64()),
    })
    write_parquet(src / f"fhvhv_tripdata_{MONTH}.parquet", {
        "pickup_datetime": pa.array([datetime(2026, 6, 1, 0, 0), datetime(2026, 6, 1, 2, 59)], pa.timestamp("us")),
        "PULocationID": pa.array([2, 2], pa.int32()),
    })
    (src / "taxi_zone_lookup.csv").write_text(
        '"LocationID","Borough","Zone","service_zone"\n'
        '1,"EWR","Newark Airport","EWR"\n'
        '2,"Queens","Jamaica Bay","Boro Zone"\n'
        '3,"Bronx","Allerton/Pelham Gardens","Boro Zone"\n'
        '264,"Unknown","N/A","N/A"\n'
        '265,"N/A","Outside of NYC","N/A"\n',
        encoding="utf-8",
    )
    return src


@pytest.fixture
def built(source, tmp_path):
    cfg = load_config()
    cfg["source_base_url"] = source.as_posix()
    cfg["zone_lookup_url"] = (source / "taxi_zone_lookup.csv").as_posix()
    paths = {
        "db_path": tmp_path / "warehouse" / "test.duckdb",
        "mart_dir": tmp_path / "mart",
        "quality_path": tmp_path / "reports" / "data_quality.json",
    }
    built_months = run([MONTH], cfg, **paths)
    mart = pd.read_parquet(paths["mart_dir"] / f"month={MONTH}" / "part.parquet")
    quality = {e["service"]: e for e in json.loads(paths["quality_path"].read_text(encoding="utf-8"))["entries"]}
    return {"cfg": cfg, "paths": paths, "built_months": built_months, "mart": mart, "quality": quality}


def cell(mart, hour, zone_id):
    row = mart[(mart["hour"] == pd.Timestamp(hour)) & (mart["zone_id"] == zone_id)]
    assert len(row) == 1
    return row.iloc[0]


def test_quality_accounts_for_every_row(built):
    yellow = built["quality"]["yellow"]
    assert yellow["rows_read"] == 9
    assert yellow["rows_outside_month"] == 2  # the 2008 row and the next-month row
    assert yellow["rows_zone_null"] == 1
    assert yellow["rows_zone_264_265"] == 2
    assert yellow["rows_zone_out_of_range"] == 0
    assert yellow["rows_in_mart"] == 4
    fhv = built["quality"]["fhv"]
    assert (fhv["rows_read"], fhv["rows_zone_null"], fhv["rows_in_mart"]) == (3, 2, 1)
    for e in built["quality"].values():
        assert e["rows_read"] == (e["rows_outside_month"] + e["rows_zone_null"] + e["rows_zone_264_265"]
                                  + e["rows_zone_out_of_range"] + e["rows_in_mart"])


def test_mart_is_a_complete_grid_with_zero_filled_hours(built):
    mart = built["mart"]
    assert len(mart) == 3 * 30 * 24  # zones 1-3 x every hour of June
    assert not mart.duplicated(["hour", "zone_id"]).any()
    assert mart["hour"].min() == pd.Timestamp("2026-06-01 00:00")
    assert mart["hour"].max() == pd.Timestamp("2026-06-30 23:00")
    # an hour with no trip at all is present with 0
    assert cell(mart, "2026-06-01 01:00", 1)["trips_total"] == 0
    # a zone with no trip in the whole month is present with 0
    assert mart.loc[mart["zone_id"] == 3, "trips_total"].sum() == 0


def test_mart_sums_the_four_services(built):
    mart = built["mart"]
    first = cell(mart, "2026-06-01 00:00", 1)
    assert (first["trips_yellow"], first["trips_green"], first["trips_total"]) == (2, 1, 3)
    zone2 = cell(mart, "2026-06-01 00:00", 2)
    assert (zone2["trips_fhv"], zone2["trips_fhvhv"], zone2["trips_total"]) == (1, 1, 2)
    assert cell(mart, "2026-06-30 23:00", 1)["trips_yellow"] == 1
    parts = mart[["trips_yellow", "trips_green", "trips_fhv", "trips_fhvhv"]].sum(axis=1)
    assert (parts == mart["trips_total"]).all()
    assert mart["trips_total"].sum() == sum(e["rows_in_mart"] for e in built["quality"].values())


def test_unknown_zones_are_excluded_from_mart_but_kept_in_fact(built):
    assert built["mart"]["zone_id"].max() <= 263
    assert set(built["mart"]["borough"]) == {"EWR", "Queens", "Bronx"}
    con = duckdb.connect(str(built["paths"]["db_path"]), read_only=True)
    try:
        unknown = con.execute("SELECT sum(trips) FROM fact_pickups_hourly WHERE zone_id IN (264, 265)").fetchone()[0]
        outside = con.execute("SELECT count(*) FROM fact_pickups_hourly WHERE hour < '2026-06-01' OR hour >= '2026-07-01'").fetchone()[0]
    finally:
        con.close()
    assert unknown == 2
    assert outside == 0


def test_dim_zone_is_exported_next_to_the_mart(built):
    zones = pd.read_csv(built["paths"]["mart_dir"].parent / "dim_zone.csv", keep_default_na=False)
    assert list(zones.columns) == ["zone_id", "borough", "zone", "service_zone"]
    assert zones["zone_id"].tolist() == [1, 2, 3, 264, 265]
    assert zones.loc[zones["zone_id"] == 2, "zone"].item() == "Jamaica Bay"


def test_existing_month_is_skipped_unless_forced(built):
    assert built["built_months"] == [MONTH]
    assert run([MONTH], built["cfg"], **built["paths"]) == []
    assert run([MONTH], built["cfg"], force=True, **built["paths"]) == [MONTH]
    rebuilt = pd.read_parquet(built["paths"]["mart_dir"] / f"month={MONTH}" / "part.parquet")
    pd.testing.assert_frame_equal(rebuilt, built["mart"])


def test_hour_calendar_flags():
    cal = hour_calendar(hours_of_months(["2025-06", "2026-07"])).set_index("hour")
    assert len(cal) == (pd.Timestamp("2026-08-01") - pd.Timestamp("2025-06-01")).days * 24
    assert list(cal.index[cal["is_dst_transition"]]) == [pd.Timestamp("2025-11-02 01:00"), pd.Timestamp("2026-03-08 02:00")]
    assert cal.loc["2025-07-04 12:00", "is_holiday"]       # Independence Day
    assert cal.loc["2025-12-25 00:00", "is_holiday"]       # Christmas
    assert cal.loc["2026-07-03 09:00", "is_holiday"]       # July 4th 2026 is a Saturday, observed on Friday
    assert not cal.loc["2026-06-10 09:00", "is_holiday"]
    assert cal.loc["2026-06-01 00:00", "day_of_week"] == 0  # a Monday
