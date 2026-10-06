"""Writes the days with an unusual citywide total to reports/unusual_days.csv.

Usage:
    python scripts/unusual_days.py

Uses the same function as the Patterns page of the app (src.analytics.unusual_days), so the
numbers quoted in the README have a file behind them.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analytics  # noqa: E402

MART = ROOT / "data" / "mart" / "demand_hourly"
OUTPUT = ROOT / "reports" / "unusual_days.csv"


def main():
    mart = pd.concat(
        [pd.read_parquet(p, columns=["hour"] + analytics.TRIP_COLUMNS) for p in sorted(MART.glob("month=*/part.parquet"))],
        ignore_index=True,
    )
    days = analytics.unusual_days(analytics.daily_totals(mart))
    days["date"] = days["date"].dt.strftime("%Y-%m-%d")
    days.round(1).to_csv(OUTPUT, index=False, lineterminator="\n")
    print(days.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
