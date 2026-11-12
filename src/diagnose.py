import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from src.config import PROCESSED

pd.set_option("display.width", 200)
df = pd.read_parquet(PROCESSED / "master.parquet")

print("== by status ==")
print(df.groupby("status").agg(
    n=("order_id", "count"),
    breach=("sla_breached", "mean"),
    avg_delay=("delivery_delay_minutes", "mean")))

dd = df[df.status.isin(["Delivered_Late", "Delivered_On_Time"])].copy()
print("\n== delay quantiles (delivered) ==")
print(dd.delivery_delay_minutes.quantile([.1, .25, .5, .75, .9, .95, .99]))

print("\n== breach rate by grace threshold (delivered) ==")
for thr in [0, 3, 5, 8, 10, 15]:
    print(f"delay > {thr:>2} min: {(dd.delivery_delay_minutes > thr).mean():.2%}")

print("\n== promised_minutes ==")
print(dd.promised_minutes.describe())

ops = dd[dd.has_ops_detail].copy()

print("\n== timeline order check (ops orders) ==")
seq = ["order_time", "pick_start", "pick_end", "assignment_time",
       "pickup_time", "actual_delivery_time"]
for a, b in zip(seq[:-1], seq[1:]):
    ok = (ops[b] >= ops[a]).mean()
    print(f"{a} <= {b}: {ok:.2%}")

print("\n== stage gaps (min) ==")
gaps = pd.DataFrame({
    "order->pick_start": (ops.pick_start - ops.order_time).dt.total_seconds() / 60,
    "pick_end->assign": (ops.assignment_time - ops.pick_end).dt.total_seconds() / 60,
    "assign->pickup": (ops.pickup_time - ops.assignment_time).dt.total_seconds() / 60,
    "pickup->delivered(orders)": (ops.actual_delivery_time - ops.pickup_time).dt.total_seconds() / 60,
})
print(gaps.describe().T[["mean", "min", "50%", "max"]])

print("\n== spearman corr with delay (ops orders) ==")
feats = ["item_count", "order_value", "promised_minutes", "hour", "dow",
         "pick_duration_minutes", "items_missing", "assignment_delay_minutes",
         "distance_km", "stockout_rate", "pending_ratio"]
print(ops[feats + ["delivery_delay_minutes"]].corr(method="spearman")
      ["delivery_delay_minutes"].drop("delivery_delay_minutes").sort_values())

print("\n== quick signal test: time split, HistGB ==")
cut = pd.Timestamp("2026-09-24")
for name, y in [("late>0", (ops.delivery_delay_minutes > 0).astype(int)),
                ("late>5", (ops.delivery_delay_minutes > 5).astype(int)),
                ("late>10", (ops.delivery_delay_minutes > 10).astype(int))]:
    tr = ops.order_time < cut
    te = ~tr
    if y[tr].nunique() < 2 or y[te].nunique() < 2:
        print(name, "skip: one class")
        continue
    m = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    m.fit(ops.loc[tr, feats], y[tr])
    auc = roc_auc_score(y[te], m.predict_proba(ops.loc[te, feats])[:, 1])
    print(f"{name}: positive rate {y.mean():.2%}, test AUC {auc:.3f}")