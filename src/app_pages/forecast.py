"""Forecast page: backtest of the XGBoost model against the last-week baseline."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics
from src.app_data import (
    BASELINE_COLOR, BLUE_RAMP, LEGEND_TOP, MAP_NO_DATA_COLOR, MAP_VIEW, PLOT_MARGIN, XGBOOST_COLOR, load_backtest,
    load_borough_metrics, load_metrics, load_zone_shapes, load_zones, neutral_ink, zone_labels,
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
        "MAE diff [95% CI]": f"{boot['mae_diff_xgb_minus_baseline']:.2f} [{boot['ci95_low']:.2f}, {boot['ci95_high']:.2f}]",
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
st.caption(f"MAE and RMSE are in trips per zone-hour. MAE diff = XGBoost minus baseline (negative is "
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

# --- Diagnostics ------------------------------------------------------------------------
month_start = pd.Timestamp(month + "-01")
in_month = (backtest["hour"] >= month_start) & (backtest["hour"] < month_start + pd.offsets.MonthBegin(1))
month_rows = backtest[in_month]
selected_rows = selected[in_month.loc[selected.index]]

st.subheader(f"Error by hour of day: {title}, {month_start:%B %Y}")
by_hour = analytics.mae_by_hour_of_day(selected_rows)
fig = go.Figure()
fig.add_scatter(x=by_hour["hour_of_day"], y=by_hour["mae_baseline"], name="Last-week baseline", mode="lines+markers",
                line=dict(color=BASELINE_COLOR, width=2, dash="dash"), marker=dict(size=8))
fig.add_scatter(x=by_hour["hour_of_day"], y=by_hour["mae_xgboost"], name="XGBoost", mode="lines+markers",
                line=dict(color=XGBOOST_COLOR, width=2), marker=dict(size=8))
fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP, height=340,
                  xaxis_title="Hour of day", yaxis_title="MAE (trips per zone-hour)")
fig.update_xaxes(dtick=1, range=[-0.5, 23.5])
fig.update_yaxes(rangemode="tozero")
st.plotly_chart(fig, width="stretch")
st.caption("Mean absolute error over the zone-hours of the selection, grouped by the hour of the day.")
with st.expander("Error by hour as a table"):
    st.dataframe(by_hour.rename(columns={"hour_of_day": "Hour of day", "mae_baseline": "MAE baseline",
                                         "mae_xgboost": "MAE XGBoost"}).style.format(
        {"MAE baseline": "{:.2f}", "MAE XGBoost": "{:.2f}"}), width="stretch", hide_index=True)

st.subheader(f"Zone map, {month_start:%B %Y}")
shapes = load_zone_shapes()
zone_names = load_zones().set_index("zone_id")
MAP_LAYOUT = dict(margin=PLOT_MARGIN, height=560,
                  map=dict(style=MAP_VIEW["map_style"], center=MAP_VIEW["center"], zoom=MAP_VIEW["zoom"]))
MAP_MARKER = dict(opacity=0.85, line=dict(width=0.5, color="#1a1a19"))
map_view = st.radio("Map view", ["Forecast for one hour", "Error by zone (month)"], horizontal=True)

if map_view == "Forecast for one hour":
    last_day = (month_start + pd.offsets.MonthEnd(0)).date()
    d1, d2, _ = st.columns([1, 1, 2])
    # keyed by month, so the date goes back to the first day when another month is chosen
    day = d1.date_input("Date", value=month_start.date(), min_value=month_start.date(), max_value=last_day,
                        key=f"map_date_{month}")
    hour_label = d2.selectbox("Hour", [f"{h:02d}:00" for h in range(24)], index=18)
    moment = pd.Timestamp(day) + pd.Timedelta(hours=int(hour_label[:2]))

    snapshot = analytics.hour_snapshot(month_rows, moment)
    snapshot["zone"] = snapshot["zone_id"].map(zone_names["zone"])
    snapshot["borough"] = snapshot["zone_id"].map(zone_names["borough"])
    # one colour scale for the whole month, ending at its largest forecast, so hours can be compared by eye
    upper, ticks, tick_labels = analytics.log_scale_ticks(month_rows["xgb"].max())
    fig = go.Figure()
    fig.add_choroplethmap(
        geojson=shapes, featureidkey="properties.location_id", locations=snapshot["zone_id"],
        z=analytics.log10_trips(snapshot["xgb"]), zmin=0, zmax=upper,
        customdata=snapshot[["zone", "borough", "xgb", "actual", "baseline", "error"]],
        colorscale=BLUE_RAMP[::-1], marker=MAP_MARKER, name="XGBoost forecast",
        colorbar=dict(title="Forecast trips", tickvals=ticks, ticktext=tick_labels),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>XGBoost %{customdata[2]:,.1f}"
                      "<br>Actual %{customdata[3]:,.0f}<br>Baseline %{customdata[4]:,.0f}"
                      "<br>XGBoost error %{customdata[5]:+,.1f}<extra></extra>",
    )
    fig.update_layout(**MAP_LAYOUT)
    st.plotly_chart(fig, width="stretch")

    kpis = analytics.snapshot_kpis(snapshot)
    k1, k2, k3 = st.columns(3)
    k1.metric("Forecast, whole city", f"{kpis['total_forecast']:,.0f}")
    k2.metric("Actual, whole city", f"{kpis['total_actual']:,.0f}")
    k3.metric("MAE per zone: XGBoost vs baseline", f"{kpis['mae_xgboost']:.1f} vs {kpis['mae_baseline']:.1f}")
    st.caption(f"XGBoost forecast of trips per zone for {moment:%Y-%m-%d %H:%M}, on a log colour scale shared by "
               "the whole month; zones below 1 trip share the darkest colour. "
               "Backtest: the model saw data up to the previous hour only. "
               "Zone boundaries: NYC TLC taxi zone shapefile, converted by scripts/fetch_zone_shapes.py.")
    with st.expander("Forecast for one hour as a table"):
        st.dataframe(
            snapshot[["zone_id", "zone", "borough", "xgb", "actual", "baseline", "error"]].rename(columns={
                "zone_id": "Zone id", "zone": "Zone", "borough": "Borough", "xgb": "XGBoost", "actual": "Actual",
                "baseline": "Baseline", "error": "XGBoost error"}).style.format(
                {"XGBoost": "{:,.1f}", "Actual": "{:,.0f}", "Baseline": "{:,.0f}", "XGBoost error": "{:+,.1f}"}),
            width="stretch", hide_index=True,
        )
else:
    zones = analytics.zone_errors(month_rows)
    zones["zone"] = zones["zone_id"].map(labels)
    stable, unstable = zones[zones["stable"]], zones[~zones["stable"]]
    hover = ("<b>%{customdata[0]}</b><br>%{customdata[1]:.1f} trips per hour<br>MAE XGBoost %{customdata[2]:.2f}"
             "<br>MAE baseline %{customdata[3]:.2f}")
    fig = go.Figure()
    fig.add_choroplethmap(
        geojson=shapes, featureidkey="properties.location_id", locations=stable["zone_id"], z=stable["wape_xgboost"],
        customdata=stable[["zone", "mean_actual", "mae_xgboost", "mae_baseline"]],
        colorscale=BLUE_RAMP[::-1], colorbar=dict(title="WAPE XGBoost (%)"), marker=MAP_MARKER,
        hovertemplate="WAPE %{z:.1f}%<br>" + hover + "<extra></extra>", name="WAPE",
    )
    fig.add_choroplethmap(
        geojson=shapes, featureidkey="properties.location_id", locations=unstable["zone_id"], z=[0] * len(unstable),
        customdata=unstable[["zone", "mean_actual", "mae_xgboost", "mae_baseline"]],
        colorscale=[[0, MAP_NO_DATA_COLOR], [1, MAP_NO_DATA_COLOR]], showscale=False,
        marker=dict(opacity=0.6, line=dict(width=0.5, color="#1a1a19")),
        hovertemplate="Too few trips for a stable percentage error<br>" + hover + "<extra></extra>", name="Too few trips",
    )
    fig.update_layout(**MAP_LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"XGBoost WAPE of every zone over the selected month. Grey: {len(unstable)} zones average fewer than "
               f"{analytics.MIN_TRIPS_FOR_WAPE} trips per hour: too few trips for a stable percentage error. "
               "Zone boundaries: NYC TLC taxi zone shapefile, converted by scripts/fetch_zone_shapes.py.")
    with st.expander("Error by zone as a table"):
        st.dataframe(
            zones.sort_values("wape_xgboost", ascending=False, na_position="last")[
                ["zone", "mean_actual", "wape_xgboost", "mae_xgboost", "mae_baseline"]].rename(columns={
                    "zone": "Zone", "mean_actual": "Mean trips per hour", "wape_xgboost": "WAPE XGBoost (%)",
                    "mae_xgboost": "MAE XGBoost", "mae_baseline": "MAE baseline"}).style.format(
                {"Mean trips per hour": "{:.1f}", "WAPE XGBoost (%)": "{:.1f}", "MAE XGBoost": "{:.2f}",
                 "MAE baseline": "{:.2f}"}, na_rep=""),
            width="stretch", hide_index=True,
        )

st.subheader(f"Largest misses: {title}, {month_start:%B %Y}")
misses = analytics.largest_misses(selected_rows, 10)
misses["zone"] = misses["zone_id"].map(labels)
st.dataframe(
    misses[["hour", "zone", "actual", "baseline", "xgb", "error"]].rename(columns={
        "hour": "Hour", "zone": "Zone", "actual": "Actual", "baseline": "Baseline", "xgb": "XGBoost",
        "error": "XGBoost error"}).style.format(
        {"Hour": "{:%Y-%m-%d %H:%M}", "Actual": "{:,.0f}", "Baseline": "{:,.0f}", "XGBoost": "{:,.1f}",
         "XGBoost error": "{:+,.1f}"}),
    width="stretch", hide_index=True,
)
st.caption(f"The {len(misses)} zone-hours of the selection where XGBoost was furthest from the actual value "
           "(error = XGBoost minus actual). No cause is attributed.")
