"""Builds the DuckDB star schema and the hourly demand mart from the public TLC parquet files.

Usage:
    python -m src.warehouse --months 2025-06:2026-07
    python -m src.warehouse --months 2026-07 --force

Each (service, month) file is read straight from its URL with DuckDB httpfs; only the pickup
timestamp and pickup zone columns are scanned. A month whose mart partition already exists is
skipped unless --force is given.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

from .config import DEFAULT_CONFIG, ROOT, load_config, month_range, parse_months
from .timeutils import hour_calendar, hours_of_months, month_bounds

SQL_DIR = ROOT / "sql"
DEFAULT_DB = ROOT / "warehouse" / "nyc_taxi.duckdb"
DEFAULT_MART = ROOT / "data" / "mart" / "demand_hourly"
DEFAULT_QUALITY = ROOT / "reports" / "data_quality.json"
HTTP_ATTEMPTS = 4
HTTP_RETRY_WAIT_SECONDS = 60


def read_sql(name):
    return (SQL_DIR / name).read_text(encoding="utf-8")


def connect(db_path, cfg):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(f"SET threads = {int(cfg['n_jobs'])}")
    con.execute(f"SET memory_limit = '{cfg['duckdb_memory_limit']}'")
    if str(cfg["source_base_url"]).startswith("http"):
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
    return con


def execute_with_retry(con, sql, params, label):
    """Runs a statement that reads a URL; returns the seconds taken by the successful attempt."""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            con.execute(sql, params)
            return round(time.perf_counter() - t0, 1)
        except (duckdb.HTTPException, duckdb.IOException) as exc:
            # The TLC CDN answers 403 for a while when it throttles a client.
            if attempt == HTTP_ATTEMPTS:
                raise
            print(f"[retry] {label}: {str(exc).splitlines()[0]} (attempt {attempt})", flush=True)
            time.sleep(HTTP_RETRY_WAIT_SECONDS * attempt)


def build_dimensions(con, cfg):
    """Rebuilds dim_service, dim_zone and dim_hour and makes sure the fact table exists."""
    con.execute(read_sql("schema.sql"))
    services = pd.DataFrame({
        "service_id": range(1, len(cfg["services"]) + 1),
        "service": list(cfg["services"]),
    })
    con.execute("CREATE OR REPLACE TABLE dim_service AS SELECT CAST(service_id AS TINYINT) AS service_id, service FROM services")
    execute_with_retry(
        con,
        """
        CREATE OR REPLACE TABLE dim_zone AS
        SELECT CAST(LocationID AS INTEGER) AS zone_id, Borough AS borough, Zone AS zone, service_zone
        FROM read_csv($url, header = true, all_varchar = true)
        ORDER BY zone_id
        """,
        {"url": str(cfg["zone_lookup_url"])},
        "taxi_zone_lookup.csv",
    )
    hours = hour_calendar(hours_of_months(month_range(cfg["months"]["start"], cfg["months"]["end"])))
    con.execute("CREATE OR REPLACE TABLE dim_hour AS SELECT * FROM hours ORDER BY hour")


def load_service_month(con, cfg, service, month):
    """Loads one (service, month) file into the fact table and returns its row accounting."""
    columns = cfg["services"][service]
    start, end = month_bounds(month)
    url = f"{str(cfg['source_base_url']).rstrip('/')}/{service}_tripdata_{month}.parquet"
    service_id = list(cfg["services"]).index(service) + 1

    sql = read_sql("load_month.sql").format(
        pickup_column=f'"{columns["pickup_column"]}"', zone_column=f'"{columns["zone_column"]}"'
    )
    seconds = execute_with_retry(con, sql, {"url": url, "month_start": start, "month_end": end}, f"{month} {service}")
    names = [d[0] for d in con.execute(read_sql("quality_month.sql")).description]
    quality = dict(zip(names, (int(v) for v in con.fetchone())))
    con.execute(read_sql("insert_fact_month.sql"), {"service_id": service_id})
    return {"service": service, "month": month, **quality, "seconds": seconds}


def write_mart_month(con, month, mart_dir):
    """Writes the complete zone x hour grid of one month to its hive partition."""
    start, end = month_bounds(month)
    con.execute(read_sql("mart_demand_hourly.sql"), {"month_start": start, "month_end": end})
    out_dir = Path(mart_dir) / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "part.parquet.tmp"
    con.execute(f"COPY mart_month TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(tmp, out_dir / "part.parquet")
    return out_dir / "part.parquet"


def update_quality(quality_path, entries, cfg):
    """Merges new (service, month) entries into the quality report, replacing older ones."""
    quality_path = Path(quality_path)
    merged = {}
    if quality_path.exists():
        for e in json.loads(quality_path.read_text(encoding="utf-8"))["entries"]:
            merged[(e["month"], e["service"])] = e
    for e in entries:
        merged[(e["month"], e["service"])] = e
    order = list(cfg["services"])
    report = {
        "generated_by": "python -m src.warehouse",
        "notes": [
            "rows_read = rows_outside_month + rows_zone_null + rows_zone_264_265 + rows_zone_out_of_range + rows_in_mart.",
            "rows_outside_month: pickup timestamp missing or not inside [first day of month, first day of next month).",
            "The zone columns count in-month rows only. Zones 264 (Unknown) and 265 (Outside of NYC) are kept in the fact table but excluded from the mart.",
            "seconds: wall time to scan the two columns over HTTP and aggregate them on the machine below.",
        ],
        "environment": {
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "duckdb": duckdb.__version__,
            "duckdb_threads": int(cfg["n_jobs"]),
            "duckdb_memory_limit": cfg["duckdb_memory_limit"],
        },
        "entries": sorted(merged.values(), key=lambda e: (e["month"], order.index(e["service"]))),
    }
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run(months, cfg, db_path=DEFAULT_DB, mart_dir=DEFAULT_MART, quality_path=DEFAULT_QUALITY, force=False):
    """Builds the requested months and returns the list of months that were (re)built."""
    todo = [m for m in months if force or not (Path(mart_dir) / f"month={m}" / "part.parquet").exists()]
    for month in months:
        if month not in todo:
            print(f"[skip] {month}: mart partition already exists (use --force to rebuild)")
    if not todo:
        return []

    con = connect(db_path, cfg)
    try:
        build_dimensions(con, cfg)
        for month in todo:
            start, end = month_bounds(month)
            con.execute("DELETE FROM fact_pickups_hourly WHERE hour >= ? AND hour < ?", [start, end])
            entries = []
            for service in cfg["services"]:
                entry = load_service_month(con, cfg, service, month)
                entries.append(entry)
                print(f"[load] {month} {service}: {entry['rows_read']:,} rows read, "
                      f"{entry['rows_in_mart']:,} in mart, {entry['seconds']}s", flush=True)
            part = write_mart_month(con, month, mart_dir)
            update_quality(quality_path, entries, cfg)
            print(f"[mart] {month}: {part.stat().st_size / 2**20:.2f} MB", flush=True)
    finally:
        con.close()
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", help="YYYY-MM or YYYY-MM:YYYY-MM (default: the range in the config)")
    ap.add_argument("--force", action="store_true", help="rebuild months whose mart partition already exists")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB file (not committed)")
    ap.add_argument("--mart-dir", default=str(DEFAULT_MART))
    ap.add_argument("--quality-path", default=str(DEFAULT_QUALITY))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    allowed = month_range(cfg["months"]["start"], cfg["months"]["end"])
    months = parse_months(args.months) if args.months else allowed
    unknown = [m for m in months if m not in allowed]
    if unknown:
        print(f"Months outside the configured range {allowed[0]}..{allowed[-1]}: {unknown}")
        return 1

    run(months, cfg, args.db, args.mart_dir, args.quality_path, args.force)
    total = sum(p.stat().st_size for p in Path(args.mart_dir).glob("month=*/part.parquet"))
    print(f"[done] mart size: {total / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
