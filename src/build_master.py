import pandas as pd
from src.data_loader import load_all
from src.config import PROCESSED


def mins(a, b):
    return (a - b).dt.total_seconds() / 60


def build() -> pd.DataFrame:
    t = load_all()

    o = t["orders"].copy()
    for c in ["order_time", "promised_time", "actual_delivery_time"]:
        o[c] = pd.to_datetime(o[c])

    p = t["picking"].copy()
    for c in ["pick_start", "pick_end"]:
        p[c] = pd.to_datetime(p[c])

    d = t["delivery"].copy()
    for c in ["assignment_time", "pickup_time", "delivery_time"]:
        d[c] = pd.to_datetime(d[c])

    # --- sanity checks ---
    print("picking order_id unique:", p.order_id.is_unique)
    print("delivery order_id unique:", d.order_id.is_unique)
    print("picking ids == delivery ids:", set(p.order_id) == set(d.order_id))

    chk = o.copy()
    chk["breach_calc"] = (chk.actual_delivery_time > chk.promised_time).astype(int)
    print("sla_breached matches time rule:", (chk.breach_calc == chk.sla_breached).mean())
    delay_calc = mins(chk.actual_delivery_time, chk.promised_time).clip(lower=0).round(1)
    print("delay matches (±0.1):", ((delay_calc - chk.delivery_delay_minutes).abs() <= 0.1).mean())

    m = o.merge(d[["order_id", "delivery_time"]], on="order_id", how="inner")
    gap = mins(m.actual_delivery_time, m.delivery_time).abs()
    print("orders vs delivery time match (<1 min):", (gap < 1).mean())

    print("pick_end >= pick_start:", (p.pick_end >= p.pick_start).mean())
    print("delivery >= pickup:", (d.delivery_time >= d.pickup_time).mean())

    # --- order-level features ---
    o["promised_minutes"] = mins(o.promised_time, o.order_time)
    o["total_minutes"] = mins(o.actual_delivery_time, o.order_time)
    o["order_date"] = o.order_time.dt.normalize()
    o["hour"] = o.order_time.dt.hour
    o["dow"] = o.order_time.dt.dayofweek
    o["is_weekend"] = (o.dow >= 5).astype(int)

    # --- store-level aggregates ---
    inv = t["inventory"]
    inv_store = inv.groupby("store_id").agg(
        stockout_rate=("stockout", lambda s: (s == "Yes").mean()),
        avg_inv_accuracy=("inventory_accuracy_pct", "mean"),
        avg_abs_variance=("stock_variance", lambda s: s.abs().mean()),
    ).reset_index()

    wf = t["workforce"]
    wf_store = wf.groupby("store_id").agg(
        headcount=("employee_id", "nunique"),
        tasks_completed=("tasks_completed", "sum"),
        tasks_pending=("tasks_pending", "sum"),
    ).reset_index()
    wf_store["pending_ratio"] = wf_store.tasks_pending / (
        wf_store.tasks_pending + wf_store.tasks_completed
    )
    wf_store = wf_store[["store_id", "headcount", "pending_ratio"]]

    # --- merge ---
    master = (
        o.merge(p, on="order_id", how="left")
         .merge(d.drop(columns=["delivery_time"]), on="order_id", how="left")
         .merge(inv_store, on="store_id", how="left")
         .merge(wf_store, on="store_id", how="left")
    )
    master["has_ops_detail"] = master.picker_id.notna()

    print("\nunique shift values:", sorted(wf["shift"].unique()))
    print("unique roles:", sorted(wf["role"].unique()))
    print("n stores:", o.store_id.nunique())
    print("date range:", o.order_time.min(), "->", o.order_time.max())
    print("breach rate all:", round(master.sla_breached.mean(), 4))
    print("breach rate with ops:", round(master[master.has_ops_detail].sla_breached.mean(), 4))
    print("breach rate no ops:", round(master[~master.has_ops_detail].sla_breached.mean(), 4))

    return master


if __name__ == "__main__":
    df = build()
    df.to_parquet(PROCESSED / "master.parquet", index=False)
    print("\nsaved master:", df.shape)