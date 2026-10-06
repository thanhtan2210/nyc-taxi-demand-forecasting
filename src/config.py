"""Loads config/pipeline.yaml and expands month ranges."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "pipeline.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def month_range(start, end):
    """Returns every month from start to end inclusive as 'YYYY-MM' strings."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def parse_months(spec):
    """Parses a CLI month spec: 'YYYY-MM' or 'YYYY-MM:YYYY-MM'."""
    start, _, end = spec.partition(":")
    return month_range(start, end or start)
