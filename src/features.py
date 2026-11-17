"""
MacroOps AI - features for SLA risk model.

Three prediction stages, same order, more information as it moves:
  stage 1  order placed     -> order + store + distance + store load
  stage 2  picking done     -> + pick time, missing items
  stage 3  rider assigned   -> + assignment delay

Target: sla_breached (delivered orders with picking/delivery detail only).
Leakage (never use): status, sla_breached, delivery_delay_minutes,
actual_delivery_time, total_minutes, pickup_time, delivery_time.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED

TARGET = "sla_breached"
PEAK_HOURS = [8, 9, 12, 13, 18, 19, 20, 21]

STAGE1 = ["item_count", "order_value", "promised_minutes", "hour", "dow", "is_weekend",
          "is_peak", "distance_km", "store_orders_60m", "load_per_head",
          "stockout_rate", "avg_inv_accuracy", "headcount", "pending_ratio"]
STAGE2 = STAGE1 + ["pick_duration_minutes", "items_missing", "pick_min_per_item", "missing_ratio"]
STAGE3 = STAGE2 + ["assignment_delay_minutes"]
STAGES = {1: STAGE1, 2: STAGE2, 3: STAGE3}
STAGE_NAMES = {1: "Order placed", 2: "Picking done", 3: "Rider assigned"}

LEAKAGE = ["status", "sla_breached", "delivery_delay_minutes", "actual_delivery_time",
           "total_minutes", "pickup_time", "delivery_time"]

LABELS = {
    "item_count": "items in basket",
    "order_value": "order value",
    "promised_minutes": "promised delivery time (min)",
    "hour": "hour of day",
    "dow": "day of week",
    "is_weekend": "weekend",
    "is_peak": "peak hour",
    "distance_km": "distance (km)",
    "store_orders_60m": "store orders in last 60 min",
    "load_per_head": "store load per 100 staff",
    "stockout_rate": "store stockout rate",
    "avg_inv_accuracy": "store inventory accuracy",
    "headcount": "store headcount",
    "pending_ratio": "store pending-task ratio",
    "pick_duration_minutes": "pick time (min)",
    "items_missing": "items missing",
    "pick_min_per_item": "pick minutes per item",
    "missing_ratio": "share of items missing",
    "assignment_delay_minutes": "rider assignment delay (min)",
}


def add_store_load(df: pd.DataFrame) -> pd.DataFrame:
    """Orders placed at the same store in the previous 60 minutes (uses all orders, past only)."""
    d = df.sort_values(["store_id", "order_time"]).reset_index(drop=True)
    t = d.order_time.values.astype("datetime64[s]").astype("int64")
    out = np.zeros(len(d))
    for _, idx in d.groupby("store_id").indices.items():
        ts = t[idx]
        left = np.searchsorted(ts, ts - 3600, side="left")
        out[idx] = np.arange(len(ts)) - left
    d["store_orders_60m"] = out
    return d


def build_features(master: pd.DataFrame) -> pd.DataFrame:
    """All orders get load features; model set is filtered afterwards."""
    d = add_store_load(master)
    d["load_per_head"] = d.store_orders_60m / d.headcount * 100
    d["is_peak"] = d.hour.isin(PEAK_HOURS).astype(int)
    d["pick_min_per_item"] = d.pick_duration_minutes / d.item_count
    d["missing_ratio"] = d.items_missing / d.item_count
    return d


def load_model_data() -> pd.DataFrame:
    m = pd.read_parquet(PROCESSED / "master.parquet")
    d = build_features(m)
    d = d[d.is_delivered & d.has_ops_detail].copy()
    return d.sort_values("order_time").reset_index(drop=True)


def time_split(d: pd.DataFrame):
    """train 1-16 Sep | valid 17-23 Sep | test 24-30 Sep."""
    train = d[d.order_time < "2026-09-17"]
    valid = d[(d.order_time >= "2026-09-17") & (d.order_time < "2026-09-24")]
    test = d[d.order_time >= "2026-09-24"]
    return train, valid, test


if __name__ == "__main__":
    d = load_model_data()
    tr, va, te = time_split(d)
    print("model rows:", len(d), "| positive rate:", round(d[TARGET].mean(), 4))
    print("train/valid/test:", len(tr), len(va), len(te))
    print("pos rate:", round(tr[TARGET].mean(), 3), round(va[TARGET].mean(), 3), round(te[TARGET].mean(), 3))
    leak = [c for s in STAGES.values() for c in s if c in LEAKAGE]
    print("leakage cols in features:", leak or "none")
    print("nulls in features:", int(d[STAGE3].isna().sum().sum()))