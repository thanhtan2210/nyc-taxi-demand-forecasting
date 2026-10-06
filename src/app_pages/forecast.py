"""Forecast page: backtest of the XGBoost model against the last-week baseline."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_data import SERIES_COLORS, load_backtest, load_borough_metrics, load_metrics

metrics = load_metrics()
backtest = load_backtest()
test_months = [name.removeprefix("test_") for name in metrics["periods"] if name.startswith("test_")]

st.title("Forecast backtest")
st.caption("One-hour-ahead backtest on historical data, not a live forecast. Each test month was "
           "predicted once by the model trained on the earlier months.")

c1, c2, c3 = st.columns([1, 1, 2])
month = c1.selectbox("Test month", test_months)
level = c2.radio("Level", ["Borough", "Zone"], horizontal=True)
if level == "Borough":
    # the busiest borough first, so the default view is the most informative one
    options = backtest.groupby("borough")["actual"].sum().sort_values(ascending=False).index.tolist()
    choice = c3.selectbox("Borough", options)
    column = "borough"
else:
    choice = c3.selectbox("Zone", sorted(backtest["zone"].unique()))
    column = "zone"

start = pd.Timestamp(month + "-01")
in_month = (backtest["hour"] >= start) & (backtest["hour"] < start + pd.offsets.MonthBegin(1))
selected = backtest[in_month & (backtest[column] == choice)]

# A borough is the sum of its zones, hour by hour.
hourly = selected.groupby("hour")[["actual", "baseline", "xgb"]].sum().reset_index()
hourly = hourly.rename(columns={"actual": "Actual", "baseline": "Last-week baseline", "xgb": "XGBoost"})

st.subheader(f"{choice}, {start:%B %Y}")
fig = px.line(
    hourly.melt(id_vars="hour", var_name="series", value_name="trips"),
    x="hour", y="trips", color="series",
    category_orders={"series": list(SERIES_COLORS)}, color_discrete_map=SERIES_COLORS,
    labels={"hour": "", "trips": "Trips per hour", "series": ""},
)
fig.update_traces(line=dict(width=2))
fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.1, x=0, title=None))
fig.update_xaxes(rangeslider_visible=True)
st.plotly_chart(fig, width="stretch")
with st.expander("Hourly values as a table"):
    st.dataframe(hourly, width="stretch", hide_index=True)

st.subheader("Accuracy by period")
rows = []
for name, period in metrics["periods"].items():
    boot = period["bootstrap"]
    rows.append({
        "Period": name.replace("test_", "Test ").replace("validation", "Validation"),
        "Zone-hours": period["n_rows"],
        "MAE baseline": period["baseline"]["mae"],
        "MAE XGBoost": period["xgboost"]["mae"],
        "MAE difference": boot["mae_diff_xgb_minus_baseline"],
        "95% CI low": boot["ci95_low"],
        "95% CI high": boot["ci95_high"],
        "RMSE baseline": period["baseline"]["rmse"],
        "RMSE XGBoost": period["xgboost"]["rmse"],
        "WAPE baseline (%)": 100 * period["baseline"]["wape"],
        "WAPE XGBoost (%)": 100 * period["xgboost"]["wape"],
    })
table = pd.DataFrame(rows)
numeric = [c for c in table.columns if c not in ("Period", "Zone-hours")]
st.dataframe(table.style.format({c: "{:.2f}" for c in numeric} | {"Zone-hours": "{:,}"}),
             width="stretch", hide_index=True)
boot = metrics["periods"]["validation"]["bootstrap"]
st.caption(f"MAE difference = XGBoost minus baseline (negative is better). Confidence interval: paired "
           f"bootstrap over days, {boot['n_bootstrap']} resamples. Validation was used to select the model, "
           "so its numbers are optimistic. From `models/metrics.json`.")

st.subheader(f"Accuracy by borough, {start:%B %Y}")
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
