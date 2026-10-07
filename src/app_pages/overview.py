"""Overview page: what the hourly demand mart contains and how it relates to the source files."""
import plotly.express as px
import streamlit as st

from src import analytics
from src.app_data import (
    BLUE_RAMP, LEGEND_TOP, MAP_VIEW, PLOT_MARGIN, SERVICE_COLORS, SERVICE_LABELS, SERVICES, load_quality_entries,
    load_zone_shapes, overview_tables,
)

tables = overview_tables()
kpis = tables["kpis"]
quality = load_quality_entries()
source = analytics.source_vs_mart(quality)
SERVICE_ORDER = [SERVICE_LABELS[s] for s in SERVICES]
COLOR_MAP = {SERVICE_LABELS[s]: SERVICE_COLORS[s] for s in SERVICES}

st.title("NYC ride demand")
st.caption("Passenger pickups per taxi zone and hour, all four TLC services combined. "
           "Source: NYC TLC trip records, aggregated by `python -m src.warehouse`.")

k1, k2, k3 = st.columns([1.2, 1.6, 0.8])
k1.metric("Zone-attributed trips", f"{kpis['total_trips']:,}")
k2.metric("Period", f"{kpis['first_hour']:%b %Y} to {kpis['last_hour']:%b %Y}")
k3.metric("Taxi zones", f"{kpis['n_zones']}")

everything = source[source["service"] == "all"].iloc[0]
gap_service, gap_reason, gap_rows = analytics.largest_gap(source)
st.markdown(
    f"{int(everything['rows_read']):,} trips in the TLC source files; {int(everything['rows_in_mart']):,} "
    f"({everything['pct_in_mart']:.1f}%) could be placed on a taxi zone. The largest part of the gap is "
    f"{gap_rows:,} {SERVICE_LABELS[gap_service]} trips {analytics.DROP_REASONS[gap_reason]}."
)
with st.expander("Source vs mart by service"):
    st.dataframe(
        source.assign(service=source["service"].map(SERVICE_LABELS).fillna("All services")).rename(columns={
            "service": "Service", "rows_read": "Source rows", "rows_outside_month": "Pickup outside the month",
            "rows_zone_null": "No pickup zone", "rows_zone_264_265": "Zone 264/265",
            "rows_zone_out_of_range": "Invalid zone id", "rows_in_mart": "Rows in mart", "pct_in_mart": "In mart (%)",
        }).style.format({"In mart (%)": "{:.3f}"}, thousands=","),
        width="stretch", hide_index=True,
    )
    st.caption("From `reports/data_quality.json`. Zone 264 is 'Unknown' and 265 is 'Outside of NYC'.")

st.subheader("Trips per day by service")
daily = tables["daily"].drop(columns="trips_total").rename(columns={f"trips_{s}": SERVICE_LABELS[s] for s in SERVICES})
fig = px.area(
    daily.melt(id_vars="date", var_name="service", value_name="trips"),
    x="date", y="trips", color="service",
    category_orders={"service": SERVICE_ORDER}, color_discrete_map=COLOR_MAP,
    labels={"date": "", "trips": "Trips per day"},
)
fig.update_traces(line=dict(width=1))
fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP)
st.plotly_chart(fig, width="stretch")
with st.expander("Trips per day as a table"):
    st.dataframe(daily, width="stretch", hide_index=True)

left, right = st.columns(2)

with left:
    st.subheader("Share of trips by service")
    share = tables["share"].assign(label=lambda d: d["service"].map(SERVICE_LABELS))
    fig = px.bar(
        share, x="share_pct", y="label", orientation="h", color="label", text="share_pct",
        category_orders={"label": SERVICE_ORDER}, color_discrete_map=COLOR_MAP,
        labels={"share_pct": "Share of zone-attributed trips (%)", "label": ""},
        hover_data={"trips": ":,", "share_pct": ":.2f", "label": False},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=40, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    null_share = analytics.fhv_null_zone_share(quality)
    st.caption(f"{SERVICE_LABELS['fhv']}: {null_share.min():.0f}-{null_share.max():.0f}% of trips per month "
               "have no pickup zone in the source files and are not counted here.")
    with st.expander("Share by service as a table"):
        st.dataframe(share[["label", "trips", "share_pct"]].rename(
            columns={"label": "Service", "trips": "Trips", "share_pct": "Share (%)"}), width="stretch", hide_index=True)

with right:
    st.subheader("Average trips by hour of day")
    profile = tables["profile"]
    fig = px.line(
        profile, x="hour_of_day", y="trips_per_hour", color="day_type",
        category_orders={"day_type": ["Weekday", "Weekend"]},
        color_discrete_map={"Weekday": "#2a78d6", "Weekend": "#eb6834"},
        labels={"hour_of_day": "Hour of day", "trips_per_hour": "Citywide trips per hour", "day_type": ""},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP)
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    st.caption("Mean over all weekdays and over all weekend days of the period.")
    with st.expander("Hourly profile as a table"):
        st.dataframe(profile.pivot(index="hour_of_day", columns="day_type", values="trips_per_hour").reset_index(),
                     width="stretch", hide_index=True)

st.subheader("Average trips per hour by zone")
zone_map = tables["zones"].assign(log_trips=lambda d: analytics.log10_trips(d["avg_trips_per_hour"]))
upper, ticks, tick_labels = analytics.log_scale_ticks(zone_map["avg_trips_per_hour"].max())
fig = px.choropleth_map(
    zone_map, geojson=load_zone_shapes(), locations="zone_id", featureidkey="properties.location_id",
    color="log_trips", color_continuous_scale=BLUE_RAMP[::-1], range_color=(0, upper),
    custom_data=["zone", "borough", "avg_trips_per_hour", "trips_total"], opacity=0.85, **MAP_VIEW,
)
fig.update_traces(
    marker_line_width=0.5, marker_line_color="#1a1a19",
    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]:,.1f} trips per hour"
                  "<br>%{customdata[3]:,} trips in the period<extra></extra>",
)
fig.update_layout(margin=PLOT_MARGIN, height=560,
                  coloraxis_colorbar=dict(title="Trips per hour", tickvals=ticks, ticktext=tick_labels))
st.plotly_chart(fig, width="stretch")
st.caption("Colour is on a log scale; zones below 1 trip per hour share the darkest colour. "
           "Zone boundaries: NYC TLC taxi zone shapefile, converted by scripts/fetch_zone_shapes.py.")
with st.expander("Trips by zone as a table"):
    st.dataframe(zone_map.sort_values("avg_trips_per_hour", ascending=False)[
        ["zone_id", "zone", "borough", "avg_trips_per_hour", "trips_total"]], width="stretch", hide_index=True)

st.subheader("Top 15 pickup zones")
top = analytics.top_zones(tables["zones"], 15).assign(label=lambda d: d["zone"] + " (" + d["borough"] + ")")
fig = px.bar(
    top.sort_values("trips_total"), x="trips_total", y="label", orientation="h",
    labels={"trips_total": "Trips in the period", "label": ""},
    color_discrete_sequence=["#2a78d6"],
)
fig.update_layout(margin=PLOT_MARGIN, height=480)
st.plotly_chart(fig, width="stretch")
with st.expander("Top zones as a table"):
    st.dataframe(top[["zone_id", "zone", "borough", "trips_total", "avg_trips_per_hour"]], width="stretch", hide_index=True)
