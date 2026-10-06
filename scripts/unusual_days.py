"""Writes the days with an unusual citywide total to reports/unusual_days.csv.

Usage:
    python -m scripts.unusual_days

Uses the same function as the Patterns page of the app (src.analytics.unusual_days), so the
numbers quoted in the README have a file behind them.
"""
import sys

import pandas as pd

from src import analytics, paths


def main():
    mart = pd.concat(
        [pd.read_parquet(p, columns=["hour"] + analytics.TRIP_COLUMNS) for p in sorted(paths.DEMAND_MART.glob("month=*/part.parquet"))],
        ignore_index=True,
    )
    days = analytics.unusual_days(analytics.daily_totals(mart))
    days["date"] = days["date"].dt.strftime("%Y-%m-%d")
    days.round(1).to_csv(paths.UNUSUAL_DAYS, index=False, lineterminator="\n")
    print(days.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
