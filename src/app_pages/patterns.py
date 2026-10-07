"""Patterns page: when and where demand happens."""
import plotly.express as px
import streamlit as st

from src import analytics
from src.app_data import (
    BOROUGH_COLORS, LEGEND_TOP, PLOT_MARGIN, SERVICE_COLORS, SERVICE_LABELS, SERVICES, load_quality_entries,
    magnitude_scale, overview_tables, patterns_tables, weekly_rhythm, zone_frame, zone_labels,
)

tables = patterns_tables()
boroughs = tables["boroughs"]
SERVICE_ORDER = [SERVICE_LABELS[s] for s in SERVICES]
SERVICE_COLOR_MAP = {SERVICE_LABELS[s]: SERVICE_COLORS[s] for s in SERVICES}

st.title("Demand patterns")
st.caption("All charts count zone-attributed pickups from the hourly mart.")

# --- 1. Weekly rhythm -------------------------------------------------------------------
st.subheader("Weekly rhythm")
c1, c2 = st.columns(2)
area = c1.selectbox("Area", ["City"] + boroughs)
service_label = c2.selectbox("Service", ["All services"] + SERVICE_ORDER)
service = next((s for s in SERVICES if SERVICE_LABELS[s] == service_label), None)
rhythm = weekly_rhythm(None if area == "City" else area, service)
fig = px.imshow(
    rhythm, aspect="auto", color_continuous_scale=magnitude_scale(),
    labels=dict(x="Hour of day", y="", color="Trips per hour"),
)
fig.update_xaxes(dtick=1)
fig.update_layout(margin=PLOT_MARGIN, height=340)
st.plotly_chart(fig, width="stretch")
st.caption(f"Mean trips per hour in {area if area != 'City' else 'the whole city'}, {service_label.lower()}, "
           "for each weekday and hour of the day.")
with st.expander("Weekly rhythm as a table"):
    st.dataframe(rhythm.style.format("{:,.0f}"), width="stretch")

# --- 2. Monthly trend by borough --------------------------------------------------------
st.subheader("Monthly trend by borough")
trend = tables["trend"]
fig = px.line(
    trend, x="month", y="trips_per_day", color="borough", markers=True,
    category_orders={"borough": list(BOROUGH_COLORS)}, color_discrete_map=BOROUGH_COLORS,
    labels={"month": "", "trips_per_day": "Trips per day (monthly mean)", "borough": ""},
)
fig.update_traces(line=dict(width=2), marker=dict(size=8))
fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP)
fig.update_yaxes(rangemode="tozero")
st.plotly_chart(fig, width="stretch")

null_share = analytics.fhv_null_zone_share(load_quality_entries()).sort_values(ascending=False)
worst, rest = null_share.head(2), null_share.iloc[2:]
st.caption(
    f"{SERVICE_LABELS['fhv']} is less complete in some months: {worst.iloc[0]:.0f}% of its trips have no pickup "
    f"zone in {worst.index[0]} and {worst.iloc[1]:.0f}% in {worst.index[1]}, against a median of "
    f"{rest.median():.0f}% in the other months (from `reports/data_quality.json`)."
)
with st.expander("Monthly trend as a table"):
    st.dataframe(trend.pivot(index="month", columns="borough", values="trips_per_day")[boroughs]
                 .style.format("{:,.0f}"), width="stretch")
st.markdown("Change against the previous month (%)")
st.dataframe(tables["month_over_month"][boroughs].iloc[1:].style.format("{:+.1f}"), width="stretch")

# --- 3. Service mix by borough ----------------------------------------------------------
st.subheader("Service mix by borough")
mix = tables["service_mix"].assign(service=lambda d: d["service"].map(SERVICE_LABELS))
fig = px.bar(
    mix, x="share_pct", y="borough", color="service", orientation="h",
    category_orders={"service": SERVICE_ORDER, "borough": boroughs}, color_discrete_map=SERVICE_COLOR_MAP,
    labels={"share_pct": "Share of the borough's trips (%)", "borough": "", "service": ""},
    hover_data={"trips": ":,", "share_pct": ":.1f"},
)
fig.update_traces(marker_line_width=0)
fig.update_layout(barmode="stack", bargap=0.35, margin=PLOT_MARGIN, legend=LEGEND_TOP, height=360)
fig.update_xaxes(range=[0, 100])
st.plotly_chart(fig, width="stretch")
with st.expander("Service mix as a table"):
    st.dataframe(mix.pivot(index="borough", columns="service", values="share_pct").loc[boroughs, SERVICE_ORDER]
                 .style.format("{:.1f}"), width="stretch")

# --- 4. Unusual days --------------------------------------------------------------------
st.subheader("Unusual days")
unusual = tables["unusual_days"]
if unusual.empty:
    st.markdown("No day is outside the thresholds.")
else:
    shown = unusual.assign(date=unusual["date"].dt.strftime("%Y-%m-%d")).rename(columns={
        "date": "Date", "weekday": "Weekday", "trips_total": "Trips", "pct_of_normal": "Total %",
        **{f"pct_{s}": f"{SERVICE_LABELS[s].split(' (')[0]} %" for s in SERVICES},
    })
    st.dataframe(shown.style.format({c: "{:.0f}" for c in shown.columns if "%" in c} | {"Trips": "{:,}"}),
                 width="stretch", hide_index=True)
st.caption(f"Days whose citywide total is below {analytics.UNUSUAL_LOW:.0%} or above {analytics.UNUSUAL_HIGH:.0%} of "
           f"normal. Normal is the median of the same weekday in the {analytics.UNUSUAL_WEEKS} weeks before and the "
           f"{analytics.UNUSUAL_WEEKS} weeks after, excluding the day itself. Each % column compares the day with "
           "its own normal level, for the total and for each service. No cause is attributed.")

# --- 5. Zone explorer -------------------------------------------------------------------
st.subheader("Zone explorer")
labels = zone_labels()
by_volume = overview_tables()["zones"].sort_values("trips_total", ascending=False)["zone_id"]
zone_of_label = {labels[z]: z for z in by_volume}
zone_label = st.selectbox("Zone", list(zone_of_label))
zone = zone_frame(zone_of_label[zone_label])
kpis = analytics.zone_kpis(zone)

k1, k2, k3 = st.columns(3)
k1.metric("Trips in the period", f"{kpis['total_trips']:,}")
k2.metric("Average trips per hour", f"{kpis['avg_trips_per_hour']:,.1f}")
k3.metric("Busiest hour of the day", f"{kpis['peak_hour_of_day']:02d}:00")

left, right = st.columns(2)
with left:
    st.markdown("**Hourly profile**")
    profile = analytics.hourly_profile(zone)
    fig = px.line(
        profile, x="hour_of_day", y="trips_per_hour", color="day_type",
        category_orders={"day_type": ["Weekday", "Weekend"]},
        color_discrete_map={"Weekday": "#2a78d6", "Weekend": "#eb6834"},
        labels={"hour_of_day": "Hour of day", "trips_per_hour": "Trips per hour", "day_type": ""},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(hovermode="x unified", margin=PLOT_MARGIN, legend=LEGEND_TOP, height=320)
    fig.update_xaxes(dtick=3)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Zone hourly profile as a table"):
        st.dataframe(profile.pivot(index="hour_of_day", columns="day_type", values="trips_per_hour").reset_index(),
                     width="stretch", hide_index=True)
with right:
    st.markdown("**Monthly trend**")
    zone_trend = analytics.monthly_trend(zone)
    fig = px.line(zone_trend, x="month", y="trips_per_day", markers=True, color_discrete_sequence=["#2a78d6"],
                  labels={"month": "", "trips_per_day": "Trips per day (monthly mean)"})
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(margin=PLOT_MARGIN, height=320)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Zone monthly trend as a table"):
        st.dataframe(zone_trend, width="stretch", hide_index=True)

st.markdown("**Service mix**")
zone_share = analytics.service_share(zone).assign(label=lambda d: d["service"].map(SERVICE_LABELS))
fig = px.bar(
    zone_share, x="share_pct", y="label", orientation="h", color="label", text="share_pct",
    category_orders={"label": SERVICE_ORDER}, color_discrete_map=SERVICE_COLOR_MAP,
    labels={"share_pct": "Share of the zone's trips (%)", "label": ""},
    hover_data={"trips": ":,", "share_pct": ":.2f", "label": False},
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
fig.update_layout(showlegend=False, margin=dict(l=10, r=40, t=10, b=10), height=260)
st.plotly_chart(fig, width="stretch")
with st.expander("Zone service mix as a table"):
    st.dataframe(zone_share[["label", "trips", "share_pct"]].rename(
        columns={"label": "Service", "trips": "Trips", "share_pct": "Share (%)"}), width="stretch", hide_index=True)
