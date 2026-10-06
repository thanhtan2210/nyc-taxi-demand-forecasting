"""Overview page: what the hourly demand mart contains."""
import plotly.express as px
import streamlit as st

from src.app_data import (
    SERVICE_COLORS, SERVICE_LABELS, SERVICES, load_fhv_null_zone_range, load_overview, load_zones,
)

LAYOUT = dict(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12, x=0, title=None))

data = load_overview()
zones = load_zones()

st.title("NYC ride demand")
st.caption("Passenger pickups per taxi zone and hour, all four TLC services combined. "
           "Source: NYC TLC trip records, aggregated by `python -m src.warehouse`.")

k1, k2, k3 = st.columns(3)
k1.metric("Total trips", f"{data['total_trips']:,}")
k2.metric("Period", f"{data['first_hour']:%b %Y} to {data['last_hour']:%b %Y}")
k3.metric("Taxi zones", f"{data['n_zones']}")

st.subheader("Trips per day by service")
daily = data["daily"].melt(id_vars="date", var_name="service", value_name="trips")
daily["service"] = daily["service"].map(SERVICE_LABELS)
fig = px.area(
    daily, x="date", y="trips", color="service",
    category_orders={"service": [SERVICE_LABELS[s] for s in SERVICES]},
    color_discrete_map={SERVICE_LABELS[s]: SERVICE_COLORS[s] for s in SERVICES},
    labels={"date": "", "trips": "Trips per day"},
)
fig.update_traces(line=dict(width=1))
fig.update_layout(hovermode="x unified", **LAYOUT)
st.plotly_chart(fig, width="stretch")
with st.expander("Daily trips as a table"):
    st.dataframe(data["daily"].rename(columns=SERVICE_LABELS), width="stretch", hide_index=True)

left, right = st.columns(2)

with left:
    st.subheader("Share of trips by service")
    totals = data["service_totals"]
    share = (100 * totals / totals.sum()).rename("share").reset_index().rename(columns={"index": "service"})
    share["trips"] = totals.to_numpy()
    share["label"] = share["service"].map(SERVICE_LABELS)
    fig = px.bar(
        share, x="share", y="label", orientation="h", color="label", text="share",
        category_orders={"label": [SERVICE_LABELS[s] for s in SERVICES]},
        color_discrete_map={SERVICE_LABELS[s]: SERVICE_COLORS[s] for s in SERVICES},
        labels={"share": "Share of trips (%)", "label": ""},
        hover_data={"trips": ":,", "share": ":.2f", "label": False},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=40, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    low, high = load_fhv_null_zone_range()
    st.caption(f"Other FHV is under-counted: {low:.0f}-{high:.0f}% of its trips per month have no pickup "
               "zone in the source files and cannot be placed on the grid "
               "(from `reports/data_quality.json`).")

with right:
    st.subheader("Average trips by hour of day")
    profile = data["profile"]
    fig = px.line(
        profile, x="hour_of_day", y="trips_total", color="day_type",
        category_orders={"day_type": ["Weekday", "Weekend"]},
        color_discrete_map={"Weekday": "#2a78d6", "Weekend": "#eb6834"},
        labels={"hour_of_day": "Hour of day", "trips_total": "Citywide trips per hour", "day_type": ""},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(hovermode="x unified", **LAYOUT)
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption("Mean over all weekdays and all weekend days of the period.")

st.subheader("Top 15 pickup zones")
top = (
    data["zone_totals"].merge(zones, on="zone_id", how="left")
    .nlargest(15, "trips_total")
    .assign(label=lambda d: d["zone"] + " (" + d["borough"] + ")")
)
fig = px.bar(
    top.sort_values("trips_total"), x="trips_total", y="label", orientation="h",
    labels={"trips_total": "Trips in the period", "label": ""},
    color_discrete_sequence=["#2a78d6"],
)
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=480)
st.plotly_chart(fig, width="stretch")
with st.expander("Top zones as a table"):
    st.dataframe(top[["zone_id", "zone", "borough", "trips_total"]], width="stretch", hide_index=True)
