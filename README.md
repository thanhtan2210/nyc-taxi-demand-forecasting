# NYC Taxi Demand Forecasting

In the next hour, roughly how many taxis does each zone of New York City need?

> **Rebuild in progress — previous version on branch `archive/before-cleanup`.**

## How it works

```text
NYC TLC parquet → BigQuery (staging → star schema) → hourly × zone features
  → XGBoost vs. "same hour last week" baseline → Streamlit (Overview, Forecast backtest) + Power BI
```

Results, run instructions and limitations will be added once the pipeline has been re-run and every number is produced by a script in this repository.

Repository: <https://github.com/thanhtan2210/nyc-taxi-demand-forecasting>
