"""Checks whether yellow trips with request_source = 'HV0003' also appear in the fhvhv file.

Usage:
    python -m scripts.double_count_check

One day (Wednesday 2026-06-10) is read straight from the TLC URLs. A yellow trip is "matched"
when an Uber (HV0003) fhvhv trip has the same pickup zone, the same drop-off zone, a pickup
time within 120 seconds and a drop-off time within 120 seconds.

Yellow trips with a null request_source are the control group: their match rate measures how
often two unrelated trips coincide by chance. Nothing is changed in the mart; the result is
written to reports/double_count_check.json.
"""
import json
import sys

import duckdb
import numpy as np

from src import paths
from src.config import load_config

DAY = "2026-06-10"
MONTH = "2026-06"
LICENSE = "HV0003"
TOLERANCE_SECONDS = 120
N_BOOTSTRAP = 1000


def bootstrap_rates(matched, total, rng):
    """Bootstrap of a match rate: resampling n binary outcomes is a binomial draw."""
    return rng.binomial(total, matched / total, N_BOOTSTRAP) / total


def interval(samples):
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main():
    cfg = load_config()
    base = str(cfg["source_base_url"]).rstrip("/")
    con = duckdb.connect()
    con.execute(f"SET threads = {int(cfg['n_jobs'])}")
    con.execute(f"SET memory_limit = '{cfg['duckdb_memory_limit']}'")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    con.execute(f"""
        CREATE TEMP TABLE yellow AS
        SELECT row_number() OVER () AS trip_id,
               tpep_pickup_datetime AS pickup, tpep_dropoff_datetime AS dropoff,
               PULocationID AS pu, DOLocationID AS do_zone, request_source
        FROM read_parquet('{base}/yellow_tripdata_{MONTH}.parquet')
        WHERE tpep_pickup_datetime >= TIMESTAMP '{DAY}' AND tpep_pickup_datetime < TIMESTAMP '{DAY}' + INTERVAL 1 DAY
    """)
    # The fhvhv window is widened by the tolerance so trips at the edges of the day can match.
    con.execute(f"""
        CREATE TEMP TABLE fhvhv AS
        SELECT pickup_datetime AS pickup, dropoff_datetime AS dropoff, PULocationID AS pu, DOLocationID AS do_zone
        FROM read_parquet('{base}/fhvhv_tripdata_{MONTH}.parquet')
        WHERE hvfhs_license_num = '{LICENSE}'
          AND pickup_datetime >= TIMESTAMP '{DAY}' - INTERVAL {TOLERANCE_SECONDS} SECOND
          AND pickup_datetime < TIMESTAMP '{DAY}' + INTERVAL 1 DAY + INTERVAL {TOLERANCE_SECONDS} SECOND
    """)
    groups = con.execute(f"""
        WITH matched AS (
            SELECT DISTINCT y.trip_id
            FROM yellow y
            JOIN fhvhv f
              ON f.pu = y.pu AND f.do_zone = y.do_zone
             AND abs(epoch(f.pickup) - epoch(y.pickup)) <= {TOLERANCE_SECONDS}
             AND abs(epoch(f.dropoff) - epoch(y.dropoff)) <= {TOLERANCE_SECONDS}
        )
        SELECT coalesce(y.request_source, 'null') AS request_source,
               count(*) AS trips,
               count(m.trip_id) AS matched
        FROM yellow y LEFT JOIN matched m USING (trip_id)
        GROUP BY 1 ORDER BY trips DESC
    """).fetchall()
    fhvhv_trips = con.execute("SELECT count(*) FROM fhvhv").fetchone()[0]
    con.close()

    counts = {name: {"trips": int(n), "matched": int(m)} for name, n, m in groups}
    rng = np.random.default_rng(int(cfg["random_state"]))
    result = {}
    samples = {}
    for name, c in counts.items():
        samples[name] = bootstrap_rates(c["matched"], c["trips"], rng)
        result[name] = {**c, "match_rate": c["matched"] / c["trips"], "ci95": interval(samples[name])}

    report = {
        "generated_by": "python -m scripts.double_count_check",
        "day": DAY,
        "match_rule": f"same pickup zone, same drop-off zone, |pickup diff| <= {TOLERANCE_SECONDS}s and |drop-off diff| <= {TOLERANCE_SECONDS}s",
        "fhvhv_license": LICENSE,
        "fhvhv_trips_in_window": int(fhvhv_trips),
        "yellow_trips": int(sum(c["trips"] for c in counts.values())),
        "yellow_by_request_source": result,
        "bootstrap": {"n": N_BOOTSTRAP, "seed": int(cfg["random_state"]), "unit": "yellow trip"},
    }
    if LICENSE in result and "null" in result:
        diff = samples[LICENSE] - samples["null"]
        report["hv0003_minus_null"] = {
            "match_rate_difference": result[LICENSE]["match_rate"] - result["null"]["match_rate"],
            "ci95": interval(diff),
        }
    paths.DOUBLE_COUNT_CHECK.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
