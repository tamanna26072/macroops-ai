"""
MacroOps AI - Copilot tools.

Each tool wraps existing, tested code (kpi.py, predict.py, rag.py) so the
Copilot never invents numbers: it can only read what these functions return.
Docstrings are the tool descriptions the LLM sees. Keep them short and clear.
"""
import json
import re
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from src import kpi
from src.config import PROCESSED

MAX_TOOL_CHARS = 2500


# ------------------------------------------------------------------ helpers
def norm_store(s) -> str:
    m = re.search(r"(\d+)", str(s or ""))
    return f"ST{int(m.group(1)):03d}" if m else ""


def norm_order(o) -> str:
    m = re.search(r"(\d+)", str(o or ""))
    return f"ORD-{int(m.group(1)):06d}" if m else ""


@lru_cache(maxsize=1)
def _master() -> pd.DataFrame:
    return kpi.load_master()


@lru_cache(maxsize=1)
def _scorecard() -> pd.DataFrame:
    return kpi.store_scorecard(_master())


@lru_cache(maxsize=1)
def _rag_store():
    from src import rag
    return rag.get_store(rag.get_embeddings())


def _slice(store_id: str = "", start: str = "", end: str = "") -> pd.DataFrame:
    df = _master()
    if store_id:
        df = df[df.store_id == store_id]
    if start:
        df = df[df.order_date >= pd.to_datetime(start)]
    if end:
        df = df[df.order_date <= pd.to_datetime(end)]
    return df


HEADLINE_KEYS = ["total_orders", "otd_pct", "sla_breach_pct", "severe_breach_pct",
                 "avg_delivery_time_min", "cancellation_pct", "avg_pick_time_min",
                 "orders_with_missing_items_pct", "item_fill_rate_pct",
                 "avg_assignment_delay_min", "sla_target_pct", "sla_gap_pct"]


def _pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d}


# ------------------------------------------------------------------ tools
@tool
def get_kpi_summary(store_id: str = "", start_date: str = "", end_date: str = "") -> dict:
    """Get delivery KPIs: on-time %, SLA breach %, severe breach %, delivery time, cancellation %,
    pick time, missing items, fill rate, rider assignment delay, and peak vs off-peak breach %.
    Optional store_id like ST095 and dates as YYYY-MM-DD. Leave blank for the whole network and month."""
    sid = norm_store(store_id)
    df = _slice(sid, start_date, end_date)
    if df.empty or not df.is_delivered.any():
        return {"error": "no delivered orders for this filter"}
    out = _pick(kpi.headline(df), HEADLINE_KEYS)
    pk = kpi.peak_vs_offpeak(df).set_index("period").breach_pct.to_dict()
    out["peak_breach_pct"] = pk.get("peak")
    out["offpeak_breach_pct"] = pk.get("off-peak")
    out["scope"] = sid or "all stores"
    out["period"] = f"{df.order_date.min():%Y-%m-%d} to {df.order_date.max():%Y-%m-%d}"
    return out


@tool
def worst_stores(n: int = 5) -> dict:
    """List the n stores with the highest SLA breach rate, with their headcount, pending task ratio,
    stockout rate and picking times. Use for questions like 'which stores are worst'."""
    n = max(1, min(int(n), 10))
    cols = ["store_id", "orders", "breach_pct", "avg_delay_min", "headcount", "pending_ratio",
            "stockout_rate", "missing_item_order_pct", "avg_pick_min", "avg_assign_delay_min"]
    sc = _scorecard().head(n)[cols]
    return {"stores": sc.to_dict("records"), "ranked_by": "breach_pct (highest first)"}


@tool
def store_report(store_id: str) -> dict:
    """Full report for one store (like ST095): breach %, rank among 100 stores (1 = worst),
    headcount, workload, stockouts, worst hours of the day, compared with the network average."""
    sid = norm_store(store_id)
    sc = _scorecard().reset_index(drop=True)
    if sid not in set(sc.store_id):
        return {"error": f"store {store_id} not found. Valid ids look like ST001 to ST100"}
    df = _slice(sid)
    h = _pick(kpi.headline(df), HEADLINE_KEYS)
    row = sc[sc.store_id == sid].iloc[0]
    hours = kpi.by_dimension(df, "hour").sort_values("breach_pct", ascending=False).head(3)
    net = kpi.headline(_master())
    return {
        "store_id": sid, **h,
        "breach_rank_of_100": int(sc.index[sc.store_id == sid][0]) + 1,
        "network_breach_pct": net["sla_breach_pct"],
        "headcount": int(row.headcount), "pending_ratio": float(row.pending_ratio),
        "stockout_rate_pct": float(row.stockout_rate),
        "worst_hours": hours[["hour", "orders", "breach_pct"]].to_dict("records"),
    }


@tool
def risky_orders(n: int = 5, store_id: str = "") -> dict:
    """List the n delivered orders with the highest predicted SLA breach risk (latest test week).
    Optional store_id. Returns order ids and risk probabilities."""
    n = max(1, min(int(n), 10))
    s = pd.read_parquet(PROCESSED / "risk_scores.parquet")
    s = s[s.split == "test"].merge(_master()[["order_id", "store_id"]], on="order_id")
    sid = norm_store(store_id)
    if sid:
        s = s[s.store_id == sid]
    if s.empty:
        return {"error": "no scored orders for this filter"}
    top = s.sort_values("p_stage3", ascending=False).head(n)
    return {"orders": [{"order_id": r.order_id, "store_id": r.store_id,
                        "risk_after_rider_assigned": round(float(r.p_stage3), 3)}
                       for r in top.itertuples()]}


@tool
def explain_order_risk(order_id: str, stage: int = 0) -> dict:
    """Explain the SLA breach risk of one order (like ORD-082529): probability, HIGH/MEDIUM/LOW,
    and the top reasons from SHAP. stage 1=order placed, 2=picking done, 3=rider assigned, 0=latest."""
    from src.predict import explain_order
    oid = norm_order(order_id)
    if not oid:
        return {"error": "order id looks like ORD-082529"}
    r = explain_order(oid, stage=int(stage) or None, top_n=5)
    if "error" in r:
        return r
    return {"order_id": r["order_id"], "store_id": r["store_id"], "stage": r["stage_name"],
            "breach_probability": r["probability"], "risk_level": r["risk_level"],
            "actually_breached": r["actual_breached"],
            "top_reasons": [f"{d['label']} = {d['value']:g} ({d['effect']})" for d in r["drivers"]]}


@tool
def search_sop(question: str) -> dict:
    """Search the operations SOP documents for procedures, policies, thresholds, escalation rules,
    compensation, incident playbooks and KPI definitions. Use for 'what should we do' questions."""
    from src import rag
    hits = rag.retrieve(question, k=3, store=_rag_store())
    if not hits:
        return {"result": "no relevant SOP found"}
    return {"excerpts": [{"source": f"{d.metadata['doc_id']} {d.metadata['title']} > {d.metadata['section']}",
                          "text": d.page_content[:700]} for d, _ in hits]}


TOOLS = [get_kpi_summary, worst_stores, store_report, risky_orders, explain_order_risk, search_sop]
TOOL_MAP = {t.name: t for t in TOOLS}


def run_tool(name: str, args: dict) -> str:
    """Run a tool safely. Always returns a JSON string, never raises."""
    try:
        if name not in TOOL_MAP:
            return json.dumps({"error": f"unknown tool {name}. Available: {list(TOOL_MAP)}"})
        result = TOOL_MAP[name].invoke(args or {})
    except Exception as e:                      # bad arguments, missing files, etc.
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"})
    text = json.dumps(result, default=str)
    return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + '..."truncated"}'