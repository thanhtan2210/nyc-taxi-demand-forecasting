"""Compares a freshly rebuilt mart partition with the one committed in the repository.

Usage:
    python -m scripts.compare_partition --month 2026-07 --rebuilt rebuilt/mart --out rebuilt/compare.json

Exits with 1 when the row count or any trips total differs.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import paths

TRIP_COLUMNS = ["trips_total", "trips_yellow", "trips_green", "trips_fhv", "trips_fhvhv"]


def summarize(path):
    frame = pd.read_parquet(path, columns=TRIP_COLUMNS)
    return {"rows": int(len(frame)), **{c: int(frame[c].sum()) for c in TRIP_COLUMNS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", required=True)
    ap.add_argument("--rebuilt", required=True, help="mart directory produced by `python -m src.warehouse --mart-dir`")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    part = Path(f"month={args.month}") / "part.parquet"
    committed = summarize(paths.DEMAND_MART / part)
    rebuilt = summarize(Path(args.rebuilt) / part)
    result = {
        "month": args.month,
        "committed": committed,
        "rebuilt": rebuilt,
        "difference": {k: rebuilt[k] - committed[k] for k in committed},
        "identical": committed == rebuilt,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
