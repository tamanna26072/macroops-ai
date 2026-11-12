"""
MacroOps AI - KPI framework.

Rules:
  - SLA KPIs (OTD, breach, delay) use DELIVERED orders only.
  - Cancellation / failure rate use ALL orders.
  - Picking KPIs use orders with picking detail (has_ops_detail).
  - Every function takes the master dataframe and returns a number or a table,
    so dashboard, API and copilot tools reuse the same logic.
"""
import pandas as pd

from src.config import PROCESSED

SLA_TARGET_PCT = 90.0        # on-time delivery target
SEVERE_DELAY_MIN = 10        # "severe" late threshold


def load_master() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "master.parquet")


def _delivered(df):
    return df[df.is_delivered]


def _ops(df):
    return df[df.has_ops_detail & df.is_delivered]


# ------------------------------------------------------------------ headline
def headline(df: pd.DataFrame) -> dict:
    dv, ops = _delivered(df), _ops(df)
    picked = ops.items_picked.sum()
    missing = ops.items_missing.sum()
    return {
        "total_orders": len(df),
        "delivered_orders": len(dv),
        "otd_pct": round((1 - dv.sla_breached.mean()) * 100, 2),
        "sla_breach_pct": round(dv.sla_breached.mean() * 100, 2),
        "severe_breach_pct": round((dv.delivery_delay_minutes > SEVERE_DELAY_MIN).mean() * 100, 2),
        "avg_delay_when_late_min": round(dv.loc[dv.sla_breached == 1, "delivery_delay_minutes"].mean(), 2),
        "p90_delay_when_late_min": round(dv.loc[dv.sla_breached == 1, "delivery_delay_minutes"].quantile(0.90), 2),
        "avg_delivery_time_min": round(dv.total_minutes.mean(), 2),
        "cancellation_pct": round((df.status == "Cancelled").mean() * 100, 2),
        "failure_pct": round((df.status == "Failed").mean() * 100, 2),
        "avg_order_value": round(df.order_value.mean(), 2),
        "avg_pick_time_min": round(ops.pick_duration_minutes.mean(), 2),
        "orders_with_missing_items_pct": round((ops.items_missing > 0).mean() * 100, 2),
        "item_fill_rate_pct": round(picked / (picked + missing) * 100, 2) if picked + missing else None,
        "avg_assignment_delay_min": round(ops.assignment_delay_minutes.mean(), 2),
        "avg_distance_km": round(ops.distance_km.mean(), 2),
        "stockout_rate_pct": round(df.drop_duplicates("store_id").stockout_rate.mean() * 100, 2),
        "inventory_accuracy_pct": round(df.drop_duplicates("store_id").avg_inv_accuracy.mean(), 2),
        "sla_target_pct": SLA_TARGET_PCT,
        "sla_gap_pct": round((1 - dv.sla_breached.mean()) * 100 - SLA_TARGET_PCT, 2),
    }


# ------------------------------------------------------------------ breakdowns
def by_dimension(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """SLA view grouped by any column: store_id, hour, dow, order_date ..."""
    dv = _delivered(df)
    g = dv.groupby(col).agg(
        orders=("order_id", "count"),
        breach_pct=("sla_breached", lambda s: s.mean() * 100),
        avg_delay_min=("delivery_delay_minutes", "mean"),
        avg_delivery_min=("total_minutes", "mean"),
    )
    return g.round(2).reset_index()


def store_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    base = by_dimension(df, "store_id")
    ops = _ops(df).groupby("store_id").agg(
        avg_pick_min=("pick_duration_minutes", "mean"),
        missing_item_order_pct=("items_missing", lambda s: (s > 0).mean() * 100),
        avg_assign_delay_min=("assignment_delay_minutes", "mean"),
    )
    static = df.drop_duplicates("store_id").set_index("store_id")[
        ["stockout_rate", "avg_inv_accuracy", "headcount", "pending_ratio"]]
    static = static.assign(stockout_rate=static.stockout_rate * 100)
    canc = df.groupby("store_id").status.apply(lambda s: (s == "Cancelled").mean() * 100).rename("cancel_pct")
    out = (base.set_index("store_id").join(ops).join(static).join(canc)
           .round(2).reset_index().sort_values("breach_pct", ascending=False))
    return out


def picker_scorecard(df: pd.DataFrame, min_orders: int = 30) -> pd.DataFrame:
    ops = _ops(df)
    g = ops.groupby("picker_id").agg(
        orders=("order_id", "count"),
        avg_pick_min=("pick_duration_minutes", "mean"),
        avg_items=("items_picked", "mean"),
        missing_pct=("items_missing", lambda s: (s > 0).mean() * 100),
        breach_pct=("sla_breached", lambda s: s.mean() * 100),
    )
    g["min_per_item"] = g.avg_pick_min / g.avg_items
    return g[g.orders >= min_orders].round(2).reset_index().sort_values("min_per_item", ascending=False)


def rider_scorecard(df: pd.DataFrame, min_orders: int = 30) -> pd.DataFrame:
    ops = _ops(df).copy()
    ops["ride_min"] = (ops.actual_delivery_time - ops.pickup_time).dt.total_seconds() / 60
    ops["kmph"] = ops.distance_km / ops.ride_min * 60
    g = ops.groupby("rider_id").agg(
        orders=("order_id", "count"),
        avg_assign_delay_min=("assignment_delay_minutes", "mean"),
        avg_distance_km=("distance_km", "mean"),
        avg_speed_kmph=("kmph", "mean"),
        breach_pct=("sla_breached", lambda s: s.mean() * 100),
    )
    return g[g.orders >= min_orders].round(2).reset_index().sort_values("breach_pct", ascending=False)


def peak_vs_offpeak(df: pd.DataFrame) -> pd.DataFrame:
    dv = _delivered(df).copy()
    dv["period"] = dv.hour.isin([8, 9, 12, 13, 18, 19, 20, 21]).map({True: "peak", False: "off-peak"})
    return by_dimension(dv, "period")


def worst_stores(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return store_scorecard(df).head(n)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    m = load_master()
    print("== HEADLINE ==")
    for k, v in headline(m).items():
        print(f"{k:32s} {v}")
    print("\n== peak vs off-peak ==")
    print(peak_vs_offpeak(m).to_string(index=False))
    print("\n== worst 5 stores ==")
    print(worst_stores(m, 5).to_string(index=False))
    print("\n== slowest 5 pickers ==")
    print(picker_scorecard(m).head(5).to_string(index=False))
    print("\n== worst 5 riders ==")
    print(rider_scorecard(m).head(5).to_string(index=False))