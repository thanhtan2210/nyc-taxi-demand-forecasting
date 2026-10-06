"""Forecast page: backtest of the XGBoost model against the last-week baseline."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics
from src.app_data import (
    BASELINE_COLOR, LEGEND_TOP, PLOT_MARGIN, XGBOOST_COLOR, load_backtest, load_borough_metrics, load_metrics,
    neutral_ink, zone_labels,
)

metrics = load_metrics()
backtest = load_backtest()
labels = zone_labels()
test_months = [name.removeprefix("test_") for name in metrics["periods"] if name.startswith("test_")]

st.title("Forecast backtest")
st.caption("One-hour-ahead backtest on historical data, not a live forecast. Each test month was "
           "predicted once by the model trained on the earlier months.")

c1, c2, c3, c4 = st.columns([1, 1, 2, 1.5])
month = c1.selectbox("Test month", test_months)
level = c2.radio("Level", ["Borough", "Zone"], horizontal=True)
if level == "Borough":
    # busiest first, so the default view is the one with the most trips
    options = backtest.groupby("borough", observed=True)["actual"].sum().sort_values(ascending=False).index.tolist()
    choice = c3.selectbox("Borough", options)
    selected = backtest[backtest["borough"] == choice]
    title = choice
else:
    # zone names repeat, so the options are the unique labels "Zone, Borough [id]"
    by_volume = backtest.groupby("zone_id")["actual"].sum().sort_values(ascending=False).index
    zone_of_label = {labels[z]: z for z in by_volume}
    title = c3.selectbox("Zone", list(zone_of_label))
    selected = backtest[backtest["zone_id"] == zone_of_label[title]]
weeks = {f"{s:%b %d} to {e - pd.Timedelta(days=1):%b %d}": (s, e) for s, e in analytics.month_weeks(month)}
start, end = weeks[c4.selectbox("Week", list(weeks))]

# A borough is the sum of its zones, hour by hour.
hourly = analytics.backtest_series(selected, start, end)

st.subheader(f"{title}, {start:%B %Y}")
fig = go.Figure()
fig.add_scatter(x=hourly["hour"], y=hourly["baseline"], name="Last-week baseline", mode="lines",
                line=dict(color=BASELINE_COLOR, width=1.5, dash="dash"), legendrank=2)
fig.add_scatter(x=hourly["hour"], y=hourly["xgb"], name="XGBoost", mode="lines",
                line=dict(color=XGBOOST_COLOR, width=1.5), legendrank=3)
# Actual is added last so it is drawn on top of the two forecasts.
fig.add_scatter(x=hourly["hour"], y=hourly["actual"], name="Actual", mode="lines",
                line=dict(color=neutral_ink(), width=3), legendrank=1)
fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP, yaxis_title="Trips per hour")
st.plotly_chart(fig, width="stretch")
with st.expander("Hourly values as a table"):
    st.dataframe(hourly.rename(columns={"actual": "Actual", "baseline": "Last-week baseline", "xgb": "XGBoost"}),
                 width="stretch", hide_index=True)

st.subheader("Accuracy by period")
rows = []
for name, period in metrics["periods"].items():
    boot = period["bootstrap"]
    rows.append({
        "Period": name.replace("test_", "Test ").replace("validation", "Validation"),
        "Zone-hours": period["n_rows"],
        "MAE baseline": period["baseline"]["mae"],
        "MAE XGBoost": period["xgboost"]["mae"],
        "MAE difference [95% CI]": f"{boot['mae_diff_xgb_minus_baseline']:.2f} [{boot['ci95_low']:.2f}, {boot['ci95_high']:.2f}]",
        "RMSE baseline": period["baseline"]["rmse"],
        "RMSE XGBoost": period["xgboost"]["rmse"],
        "WAPE baseline (%)": 100 * period["baseline"]["wape"],
        "WAPE XGBoost (%)": 100 * period["xgboost"]["wape"],
    })
table = pd.DataFrame(rows)
numeric = table.select_dtypes("float").columns
st.dataframe(table.style.format({c: "{:.2f}" for c in numeric} | {"Zone-hours": "{:,}"}),
             width="stretch", hide_index=True)
boot = metrics["periods"]["validation"]["bootstrap"]
st.caption(f"MAE and RMSE are in trips per zone-hour. MAE difference = XGBoost minus baseline (negative is "
           f"better). Confidence interval: paired bootstrap over days, {boot['n_bootstrap']} resamples. "
           "Validation was used to select the model, so its numbers are optimistic. From `models/metrics.json`.")

st.subheader(f"Accuracy by borough, {pd.Timestamp(month + '-01'):%B %Y}")
boroughs = load_borough_metrics()
boroughs = boroughs[boroughs["period"] == f"test_{month}"].drop(columns="period")
boroughs = boroughs.assign(wape_baseline=100 * boroughs["wape_baseline"], wape_xgboost=100 * boroughs["wape_xgboost"])
boroughs = boroughs.rename(columns={
    "borough": "Borough", "n_zones": "Zones", "n_rows": "Zone-hours", "mean_actual": "Mean trips per zone-hour",
    "mae_baseline": "MAE baseline", "mae_xgboost": "MAE XGBoost",
    "wape_baseline": "WAPE baseline (%)", "wape_xgboost": "WAPE XGBoost (%)",
})
st.dataframe(
    boroughs.style.format({"Zone-hours": "{:,}", "Mean trips per zone-hour": "{:.1f}", "MAE baseline": "{:.2f}",
                           "MAE XGBoost": "{:.2f}", "WAPE baseline (%)": "{:.1f}", "WAPE XGBoost (%)": "{:.1f}"}),
    width="stretch", hide_index=True,
)
st.caption("From `reports/metrics_by_borough.csv`.")
