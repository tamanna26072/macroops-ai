"""
MacroOps AI - Operations Dashboard (Streamlit).

Run from repo root:
    streamlit run app/dashboard.py
Needs data/processed/master.parquet  (python -m src.build_master)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.kpi import (SLA_TARGET_PCT, by_dimension, headline, load_master,
                     peak_vs_offpeak, picker_scorecard, rider_scorecard,
                     store_scorecard)

st.set_page_config(page_title="MacroOps AI", page_icon="📦", layout="wide")

# ------------------------------------------------------------------ theme
ACCENT = "#00D4AA"
PALETTE = ["#00D4AA", "#4C8DFF", "#FFB020", "#FF5C5C", "#A78BFA", "#38BDF8"]
STATUS_COLORS = {"Delivered_On_Time": "#00D4AA", "Delivered_Late": "#FFB020",
                 "Cancelled": "#FF5C5C", "Failed": "#A83232"}
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = PALETTE

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
h1 {font-weight: 800 !important; letter-spacing: -0.5px;}
h2, h3 {font-weight: 700 !important;}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #131A26 0%, #182131 100%);
    border: 1px solid #243047; border-radius: 14px;
    padding: 16px 20px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
div[data-testid="stMetricLabel"] p {color: #8FA0BA; font-size: 0.82rem; text-transform: uppercase; letter-spacing: .04em;}
div[data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 750;}
section[data-testid="stSidebar"] {border-right: 1px solid #1B2433;}
.hero {background: linear-gradient(90deg, rgba(0,212,170,0.16), rgba(76,141,255,0.06) 60%, transparent);
       border-left: 4px solid #00D4AA; border-radius: 6px; padding: 14px 20px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 4px 0 0 0; color: #8FA0BA; font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BREACH_TARGET = 100 - SLA_TARGET_PCT


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner="Loading master table...")
def get_master() -> pd.DataFrame:
    return load_master()


def show_chart(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title_font=dict(size=16), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#1E2738")
    fig.update_yaxes(gridcolor="#1E2738")
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # older streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    cfg = {}
    if "breach_pct" in df.columns:
        cfg["breach_pct"] = st.column_config.ProgressColumn(
            "breach %", min_value=0, max_value=100, format="%.1f%%")
    try:
        st.dataframe(df, width="stretch", hide_index=True, column_config=cfg)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta, inv) in zip(cols, items):
        col.metric(label, value, delta, delta_color="inverse" if inv else "normal")


# ------------------------------------------------------------------ data + filters
try:
    master = get_master()
except FileNotFoundError:
    st.error("master.parquet not found. Run: python -m src.build_master")
    st.stop()

st.sidebar.title("📦 MacroOps AI")
page = st.sidebar.radio("Page", ["Overview", "Store drill-down", "Peak hours & trends",
                                 "Picker & Rider performance"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
dmin, dmax = master.order_date.min().date(), master.order_date.max().date()
picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(picked, (tuple, list)) and len(picked) == 2:
    d0, d1 = picked
else:
    d0 = d1 = picked[0] if isinstance(picked, (tuple, list)) else picked

stores_all = sorted(master.store_id.unique())
sel_stores = st.sidebar.multiselect("Stores (empty = all)", stores_all)

df = master[(master.order_date.dt.date >= d0) & (master.order_date.dt.date <= d1)]
if sel_stores:
    df = df[df.store_id.isin(sel_stores)]

if df.empty or not df.is_delivered.any():
    st.warning("No delivered orders for this filter.")
    st.stop()

H = headline(df)
st.sidebar.caption(f"{H['total_orders']:,} orders  |  SLA target {SLA_TARGET_PCT:.0f}% on-time")

# ================================================================== OVERVIEW
if page == "Overview":
    header("Operations Overview",
           f"{d0:%d %b} - {d1:%d %b %Y}  |  {H['total_orders']:,} orders  |  SLA metrics = delivered orders only")

    kpi_row([
        ("On-time delivery", f"{H['otd_pct']}%", f"{H['sla_gap_pct']} pts vs target", False),
        ("SLA breach", f"{H['sla_breach_pct']}%", None, True),
        ("Severe breach (>10 min)", f"{H['severe_breach_pct']}%", None, True),
        ("Avg delivery time", f"{H['avg_delivery_time_min']} min", None, True),
    ])
    kpi_row([
        ("Cancellation", f"{H['cancellation_pct']}%", None, True),
        ("Avg pick time", f"{H['avg_pick_time_min']} min", None, True),
        ("Item fill rate", f"{H['item_fill_rate_pct']}%", None, False),
        ("Avg rider assignment delay", f"{H['avg_assignment_delay_min']} min", None, True),
    ])

    c1, c2 = st.columns([2, 1])
    with c1:
        daily = by_dimension(df, "order_date")
        fig = px.line(daily, x="order_date", y="breach_pct", markers=True,
                      title="Daily SLA breach %", labels={"breach_pct": "breach %", "order_date": ""})
        fig.add_hline(y=BREACH_TARGET, line_dash="dash", line_color="#00D4AA",
                      annotation_text=f"target {BREACH_TARGET:.0f}%")
        show_chart(fig)
    with c2:
        stat = df.status.value_counts().reset_index()
        stat.columns = ["status", "orders"]
        fig = px.bar(stat, x="orders", y="status", orientation="h", color="status",
                     color_discrete_map=STATUS_COLORS, text="orders", title="Order status")
        fig.update_layout(showlegend=False, yaxis_title=None)
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        show_chart(fig)

    hourly = by_dimension(df, "hour")
    fig = px.bar(hourly, x="hour", y="breach_pct", title="SLA breach % by hour of day",
                 labels={"breach_pct": "breach %"})
    fig.add_hline(y=BREACH_TARGET, line_dash="dash", line_color="#00D4AA")
    show_chart(fig)

    st.subheader("Needs attention")
    sc = store_scorecard(df)
    worst = sc.head(5)[["store_id", "orders", "breach_pct", "avg_delay_min",
                        "headcount", "pending_ratio", "missing_item_order_pct"]]
    show_table(worst)

# ================================================================== STORES
elif page == "Store drill-down":
    header("Store drill-down", "Find weak stores and see why: staffing, picking, stockouts")
    sc = store_scorecard(df)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sc.head(15), x="store_id", y="breach_pct", title="Top 15 stores by breach %",
                     hover_data=["orders", "headcount", "avg_pick_min"])
        fig.add_hline(y=BREACH_TARGET, line_dash="dash", line_color="#00D4AA")
        show_chart(fig)
    with c2:
        fig = px.scatter(sc, x="headcount", y="breach_pct", size="orders", hover_name="store_id",
                         title="Staffing vs breach % (bubble = order volume)",
                         labels={"breach_pct": "breach %"})
        show_chart(fig)

    st.subheader("Store scorecard")
    show_table(sc)

    st.subheader("Single store")
    store = st.selectbox("Store", sorted(sc.store_id), index=0)
    sdf = df[df.store_id == store]
    SH = headline(sdf)
    kpi_row([
        ("On-time delivery", f"{SH['otd_pct']}%", f"{SH['sla_gap_pct']} pts vs target", False),
        ("Avg pick time", f"{SH['avg_pick_time_min']} min", None, True),
        ("Orders w/ missing items", f"{SH['orders_with_missing_items_pct']}%", None, True),
        ("Stockout rate", f"{SH['stockout_rate_pct']}%", None, True),
    ])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_dimension(sdf, "hour"), x="hour", y="breach_pct",
                     title=f"{store}: breach % by hour", labels={"breach_pct": "breach %"})
        show_chart(fig)
    with c2:
        fig = px.line(by_dimension(sdf, "order_date"), x="order_date", y="breach_pct", markers=True,
                      title=f"{store}: daily breach %", labels={"breach_pct": "breach %"})
        show_chart(fig)

# ================================================================== PEAK / TRENDS
elif page == "Peak hours & trends":
    header("Peak hours & trends", "When do SLAs break? Weekday x hour, volume vs breach")
    dv = df[df.is_delivered]

    heat = dv.pivot_table(index="dow", columns="hour", values="sla_breached", aggfunc="mean") * 100
    heat.index = [DOW[i] for i in heat.index]
    fig = px.imshow(heat.round(1), aspect="auto", color_continuous_scale="YlOrRd",
                    title="SLA breach % by weekday x hour", labels=dict(color="breach %", x="hour"))
    show_chart(fig)

    hourly = by_dimension(df, "hour")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hourly.hour, y=hourly.orders, name="orders", opacity=0.5), secondary_y=False)
    fig.add_trace(go.Scatter(x=hourly.hour, y=hourly.breach_pct, name="breach %", mode="lines+markers"),
                  secondary_y=True)
    fig.update_layout(title="Order volume vs breach % by hour")
    fig.update_yaxes(title_text="orders", secondary_y=False)
    fig.update_yaxes(title_text="breach %", secondary_y=True)
    show_chart(fig)

    st.subheader("Peak vs off-peak")
    st.caption("Peak = 8-9, 12-13, 18-21")
    show_table(peak_vs_offpeak(df))

    dow_df = by_dimension(df, "dow")
    dow_df["day"] = dow_df.dow.map(lambda i: DOW[i])
    show_chart(px.bar(dow_df, x="day", y="breach_pct", title="Breach % by weekday",
                      labels={"breach_pct": "breach %"}))

# ================================================================== PICKER / RIDER
else:
    header("Picker & Rider performance", "Based on orders with picking / delivery detail")
    min_orders = st.slider("Minimum orders per person", 10, 100, 30, step=5)
    tab1, tab2 = st.tabs(["Pickers", "Riders"])

    with tab1:
        pk = picker_scorecard(df, min_orders)
        if pk.empty:
            st.info("No pickers meet the minimum for this filter.")
        else:
            fig = px.scatter(pk, x="min_per_item", y="breach_pct", size="orders", hover_name="picker_id",
                             color="missing_pct", color_continuous_scale="OrRd",
                             title="Picker speed (min per item) vs breach %",
                             labels={"breach_pct": "breach %", "missing_pct": "missing %"})
            show_chart(fig)
            st.subheader("Slowest pickers")
            show_table(pk.head(20))

    with tab2:
        rd = rider_scorecard(df, min_orders)
        if rd.empty:
            st.info("No riders meet the minimum for this filter.")
        else:
            fig = px.scatter(rd, x="avg_speed_kmph", y="breach_pct", size="orders", hover_name="rider_id",
                             color="avg_assign_delay_min", color_continuous_scale="OrRd",
                             title="Rider speed vs breach %",
                             labels={"breach_pct": "breach %"})
            show_chart(fig)
            st.subheader("Highest-breach riders")
            show_table(rd.head(20))