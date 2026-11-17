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

from src.features import STAGE_NAMES
from src.kpi import (SLA_TARGET_PCT, by_dimension, headline, load_master,
                     peak_vs_offpeak, picker_scorecard, rider_scorecard,
                     store_scorecard)

from src.predict import explain_order, get_threshold, models_ready, risk_level

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
@st.cache_data(show_spinner="Loading risk scores...")
def get_scores() -> pd.DataFrame:
    from src.config import PROCESSED
    return pd.read_parquet(PROCESSED / "risk_scores.parquet")


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
    if "risk_pct" in df.columns:
        cfg["risk_pct"] = st.column_config.ProgressColumn(
            "risk %", min_value=0, max_value=100, format="%.0f%%")
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
                                 "Picker & Rider performance", "At-risk orders", "AI Copilot"])

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

# ================================================================== AT-RISK ORDERS
elif page == "At-risk orders":
    header("At-risk orders", "SLA breach prediction: XGBoost + SHAP. Risk updates as the order moves through the pipeline")
    if not models_ready():
        st.info("Model not trained yet. Run: python -m src.train_model")
        st.stop()

    c1, c2 = st.columns([2, 1])
    stage = c1.radio("Prediction stage", [1, 2, 3], index=2, horizontal=True,
                     format_func=lambda s: f"{s}. {STAGE_NAMES[s]}")
    split = c2.selectbox("Data split", ["test", "valid", "train", "all"], index=0,
                         format_func=lambda x: {"test": "Test week (24-30 Sep)", "valid": "Validation week (17-23 Sep)",
                                                "train": "Train period (1-16 Sep)", "all": "All"}[x])
    thr = get_threshold(stage)

    scores = get_scores()
    base = df[df.has_ops_detail & df.is_delivered][["order_id", "store_id", "order_time", "sla_breached"]]
    d = base.merge(scores[["order_id", "split", "p_stage1", "p_stage2", "p_stage3"]], on="order_id")
    if split != "all":
        d = d[d.split == split]
    if d.empty:
        st.warning("No scored orders for this filter and split.")
        st.stop()

    d = d.assign(risk=d[f"p_stage{stage}"])
    d["level"] = d.risk.apply(lambda p: risk_level(p, thr))
    flagged = d[d.risk >= thr]
    caught = flagged.sla_breached.sum()
    total_breach = d.sla_breached.sum()

    kpi_row([
        ("Orders scored", f"{len(d):,}", None, False),
        ("Flagged HIGH risk", f"{len(flagged):,}", f"{len(flagged) / len(d):.0%} of orders", True),
        ("Flagged that really breached", f"{(caught / len(flagged)):.0%}" if len(flagged) else "-", None, False),
        ("Breaches caught", f"{(caught / total_breach):.0%}" if total_breach else "-", None, False),
    ])
    st.caption(f"Alert threshold {thr:.2f} (tuned for 70% recall on validation week). "
               "Backtest on history: 'really breached' is known after delivery.")

    c1, c2 = st.columns(2)
    with c1:
        hist = d.assign(outcome=d.sla_breached.map({0: "On time", 1: "Breached"}))
        fig = px.histogram(hist, x="risk", color="outcome", nbins=40, barmode="overlay", opacity=0.75,
                           color_discrete_map={"On time": "#00D4AA", "Breached": "#FF5C5C"},
                           title="Risk score: breached vs on-time orders", labels={"risk": "predicted risk"})
        fig.add_vline(x=thr, line_dash="dash", line_color="#FFB020", annotation_text="alert threshold")
        show_chart(fig)
    with c2:
        by_store = flagged.groupby("store_id").size().sort_values(ascending=False).head(10).reset_index()
        by_store.columns = ["store_id", "flagged_orders"]
        show_chart(px.bar(by_store, x="store_id", y="flagged_orders", title="Stores with most flagged orders"))

    st.subheader("Highest-risk orders")
    top = d.sort_values("risk", ascending=False).head(50).copy()
    top["risk_pct"] = (top.risk * 100).round(1)
    top["actual"] = top.sla_breached.map({0: "on time", 1: "breached"})
    show_table(top[["order_id", "store_id", "order_time", "risk_pct", "level", "actual"]])

    st.subheader("Why is this order at risk?")
    oid = st.selectbox("Order", top.order_id.tolist())
    with st.spinner("Computing SHAP explanation..."):
        ex = explain_order(oid, stage=stage)
    if "error" in ex:
        st.warning(ex["error"])
    else:
        kpi_row([
            ("Breach probability", f"{ex['probability']:.0%}", None, True),
            ("Risk level", ex["risk_level"], None, True),
            ("Store", ex["store_id"], None, False),
            ("Actual outcome", "breached" if ex["actual_breached"] else "on time", None, False),
        ])
        st.info(ex["summary"])
        e1, e2 = st.columns(2)
        with e1:
            dr = pd.DataFrame(ex["drivers"])
            dr["name"] = dr.label + " = " + dr.value.astype(str)
            dr["dir"] = dr.shap.apply(lambda v: "raises risk" if v > 0 else "lowers risk")
            fig = px.bar(dr.sort_values("shap"), x="shap", y="name", orientation="h", color="dir",
                         color_discrete_map={"raises risk": "#FF5C5C", "lowers risk": "#00D4AA"},
                         title="Top drivers (SHAP)", labels={"shap": "impact on risk", "name": ""})
            show_chart(fig)
        with e2:
            row = d[d.order_id == oid].iloc[0]
            j = pd.DataFrame({"stage": [STAGE_NAMES[s] for s in (1, 2, 3)],
                              "risk": [row.p_stage1, row.p_stage2, row.p_stage3]})
            fig = px.line(j, x="stage", y="risk", markers=True, title="Risk as the order progresses",
                          range_y=[0, 1], labels={"risk": "breach probability"})
            show_chart(fig)

# ================================================================== AI COPILOT
elif page == "AI Copilot":
    header("AI Operations Copilot", "Ask about KPIs, stores, order risk and SOPs. Numbers come from tools, never guessed")
    st.caption("Copilot reads the full month and all stores. Sidebar filters do not apply here. Model: qwen2.5:3b via Ollama.")

    if "copilot" not in st.session_state:
        try:
            from src.copilot import Copilot
            with st.spinner("Starting Copilot..."):
                st.session_state.copilot = Copilot()
        except Exception as e:
            st.error(f"Could not start Copilot: {e}")
            st.info("Check Ollama is running (ollama list) and langchain-ollama is installed.")
            st.stop()
        st.session_state.chat = []
    cp = st.session_state.copilot

    ideas = ["What is our on-time delivery rate?", "Which stores have the highest breach rate?",
             "What should the dispatcher do if rider assignment takes 12 minutes?",
             "Show me the 3 riskiest orders"]
    cols = st.columns(len(ideas))
    for c, q in zip(cols, ideas):
        if c.button(q, key=f"idea_{q}"):
            st.session_state.pending_q = q

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                with st.expander(f"Tools used ({len(m['tools'])})"):
                    for t in m["tools"]:
                        st.code(f"{t['name']}({t['args']})\n{t['result'][:600]}", language="json")

    typed = st.chat_input("Ask the Copilot...")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    r = cp.chat(question)
                st.markdown(r["answer"])
                if r["tools"]:
                    with st.expander(f"Tools used ({len(r['tools'])})"):
                        for t in r["tools"]:
                            st.code(f"{t['name']}({t['args']})\n{t['result'][:600]}", language="json")
                st.caption(f"{r['seconds']}s")
                st.session_state.chat.append({"role": "assistant", "content": r["answer"], "tools": r["tools"]})
            except Exception as e:
                st.error(f"Copilot error: {e}")
                st.info("Is Ollama running? Try: ollama list")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Copilot memory")
        f = cp.memory.facts
        st.write(f"Store: **{f['store_id'] or 'not set'}**")
        st.write(f"Role: **{f['role'] or 'not set'}**")
        for n in f["notes"]:
            st.caption(f"Note: {n}")
        st.caption("Try: 'My store is ST095' then ask about 'my store'.")
        if st.button("Clear chat"):
            cp.reset_chat()
            st.session_state.chat = []
            st.rerun()
        if st.button("Forget memory"):
            cp.memory.update_from("forget everything")
            st.rerun()

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