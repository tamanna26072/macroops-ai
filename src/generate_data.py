"""
MacroOps AI - synthetic data generator v2.

Same 5 CSV schemas as v1, but delays now have real causes:
  - queue congestion (store load vs headcount, peak hours, promo day)
  - stockouts -> missing items -> longer picking
  - slow pickers / slow riders / weak stores
  - rider assignment delay, distance, rain days
Promised time uses only info known at order time (items, distance),
so congestion / stockouts / rain are the "surprises" that cause SLA breach.

Run from repo root:  python -m src.generate_data
"""
import shutil

import numpy as np
import pandas as pd

from src.config import RAW, ROOT

SEED = 42
N_ORDERS = 100_000
N_STORES = 100
N_OPS = 25_000            # orders with picking + delivery detail
N_WORKFORCE = 10_000
SKU_UNIVERSE = 1_000
SKU_PER_STORE = 400       # 100 x 400 = 40,000 inventory rows
PICKERS_PER_STORE = 12
RIDERS_PER_STORE = 15
START = pd.Timestamp("2026-09-01")
DAYS = 30

# tuning knobs
PROMISE_BUFFER_MIN = 9.0  # raise -> fewer late orders
RAIN_DAYS = [11, 20]      # 12 Sep, 21 Sep
PROMO_DAY = 18            # 19 Sep, +50% volume

HOUR_W = np.array([0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.8, 1.4, 2.2, 2.4, 1.8, 1.7,
                   2.2, 2.4, 1.8, 1.4, 1.6, 2.2, 3.0, 3.4, 3.4, 3.0, 2.0, 1.0])
PEAK_HOURS = {8, 9, 12, 13, 18, 19, 20, 21}

rng = np.random.default_rng(SEED)


def secs(minutes):
    return np.round(np.asarray(minutes) * 60).astype(int)


def fmt(ts):
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ stores
def make_stores():
    ids = np.array([f"ST{i:03d}" for i in range(1, N_STORES + 1)])
    share = rng.lognormal(0, 0.35, N_STORES)
    share /= share.sum()

    staff_ratio = rng.normal(1.0, 0.10, N_STORES).clip(0.75, 1.25)
    weak = rng.choice(N_STORES, 12, replace=False)          # understaffed stores
    staff_ratio[weak] = rng.uniform(0.55, 0.70, 12)

    p_emp = share * staff_ratio
    p_emp /= p_emp.sum()
    emp_store = rng.choice(N_STORES, N_WORKFORCE, p=p_emp)
    headcount = np.bincount(emp_store, minlength=N_STORES)

    stockout = rng.lognormal(np.log(0.03), 0.30, N_STORES).clip(0.01, 0.07)
    bad_inv = rng.choice(N_STORES, 10, replace=False)       # stockout-prone stores
    stockout[bad_inv] = rng.uniform(0.10, 0.16, 10)

    stress = (share / share.sum()) / (headcount / headcount.sum())
    store_eff = rng.lognormal(0, 0.08, N_STORES)

    pick_skill = rng.lognormal(0, 0.18, (N_STORES, PICKERS_PER_STORE))
    rider_speed = rng.lognormal(0, 0.12, (N_STORES, RIDERS_PER_STORE))

    return dict(ids=ids, share=share, headcount=headcount, emp_store=emp_store,
                stockout=stockout, stress=stress, store_eff=store_eff,
                pick_skill=pick_skill, rider_speed=rider_speed)


# ------------------------------------------------------------------ orders + chain
def make_orders_and_chain(S):
    n = N_ORDERS
    day_w = np.ones(DAYS)
    for d in range(DAYS):
        if (START + pd.Timedelta(days=d)).dayofweek >= 5:
            day_w[d] *= 1.25
    day_w[PROMO_DAY] *= 1.5

    day = rng.choice(DAYS, n, p=day_w / day_w.sum())
    hour = rng.choice(24, n, p=HOUR_W / HOUR_W.sum())
    minute = rng.integers(0, 60, n)
    order_time = (START + pd.to_timedelta(day, unit="D")
                  + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m"))

    order = np.argsort(order_time.values, kind="stable")
    day, hour, minute, order_time = day[order], hour[order], minute[order], order_time[order]

    store_idx = rng.choice(N_STORES, n, p=S["share"])
    items = (1 + rng.poisson(3.2, n)).clip(1, 15)
    order_value = np.round(items * rng.lognormal(np.log(110), 0.30, n), 2)
    dist = rng.lognormal(np.log(2.4), 0.55, n).clip(0.4, 10).round(2)

    peak = np.isin(hour, list(PEAK_HOURS)).astype(float)
    rain = np.isin(day, RAIN_DAYS).astype(float)

    # congestion: orders in same store-hour vs store capacity
    key = store_idx * (DAYS * 24) + day * 24 + hour
    cnt = np.bincount(key, minlength=N_STORES * DAYS * 24)[key]
    cap = 0.03 * S["headcount"][store_idx]
    cong = cnt / cap

    picker_j = rng.integers(0, PICKERS_PER_STORE, n)
    rider_j = rng.integers(0, RIDERS_PER_STORE, n)
    p_skill = S["pick_skill"][store_idx, picker_j]
    r_speed = S["rider_speed"][store_idx, rider_j]

    # stage 1: queue wait
    wait = (1.5 + 5.0 * np.clip(cong - 0.7, 0, None) ** 1.3
            + rng.exponential(0.8, n)).clip(1, 25)

    # stage 2: picking
    missing = rng.binomial(items, np.clip(S["stockout"][store_idx] * 1.5, 0, 0.5))
    pick = ((1.5 + 1.1 * items) * p_skill * S["store_eff"][store_idx]
            * (1 + 0.15 * np.clip(cong - 0.8, 0, None))
            + 2.0 * missing + rng.gamma(4, 0.3, n))

    # stage 3: rider assignment (wait after pick done)
    a_mean = 1.2 + 1.8 * peak + 2.2 * np.clip(cong - 0.8, 0, None) + 2.0 * rain
    assign = rng.gamma(2, a_mean / 2).clip(0.5, 25)

    # stage 4: rider to store, then delivery
    to_store = rng.gamma(4, 1.0, n).clip(1.5, 15) * (1 + 0.3 * rain)
    speed = 17.0 * r_speed * (1 - 0.15 * peak) * (1 - 0.38 * rain)
    travel = dist / speed * 60 + 1.5 + rng.gamma(3, 0.4, n)

    w_s, pk_s, as_s = secs(wait), secs(pick), secs(assign)
    ts_s, tr_s = secs(to_store), secs(travel)
    t_pick_start = order_time + pd.to_timedelta(w_s, unit="s")
    t_pick_end = t_pick_start + pd.to_timedelta(pk_s, unit="s")
    t_assign = t_pick_end + pd.to_timedelta(as_s, unit="s")
    t_pickup = t_assign + pd.to_timedelta(ts_s, unit="s")
    t_deliver = t_pickup + pd.to_timedelta(tr_s, unit="s")

    # promise: only what is known at order time
    nominal = 1.5 + (1.5 + 1.1 * items + 1.2) + 2.5 + 4.0 + (dist / 17 * 60 + 1.5 + 1.2)
    promised_min = (nominal + PROMISE_BUFFER_MIN + rng.normal(0, 1.2, n)).clip(20, 60)
    promised = order_time + pd.to_timedelta(secs(promised_min), unit="s")

    # status
    p_cancel = np.clip(0.008 + 0.003 * wait, 0, 0.1)
    p_fail = 0.004 + 0.01 * (missing >= 1)
    u = rng.random(n)
    status = np.where(u < p_cancel, "Cancelled",
             np.where(u < p_cancel + p_fail, "Failed", "Delivered"))
    delivered = status == "Delivered"

    late = (t_deliver > promised) & delivered
    status = np.where(delivered, np.where(late, "Delivered_Late", "Delivered_On_Time"), status)

    delay = np.where(delivered,
                     np.clip((t_deliver - promised).total_seconds() / 60, 0, None), 0.0).round(1)
    actual = pd.Series(t_deliver).where(delivered)

    orders = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
        "store_id": S["ids"][store_idx],
        "order_time": fmt(pd.Series(order_time)),
        "item_count": items,
        "order_value": order_value,
        "promised_time": fmt(pd.Series(promised)),
        "actual_delivery_time": fmt(actual),
        "status": status,
        "sla_breached": late.astype(int),
        "delivery_delay_minutes": delay,
    })

    chain = dict(store_idx=store_idx, picker_j=picker_j, rider_j=rider_j, items=items,
                 missing=missing, dist=dist, delivered=delivered,
                 t_pick_start=pd.Series(t_pick_start), t_pick_end=pd.Series(t_pick_end),
                 t_assign=pd.Series(t_assign), t_pickup=pd.Series(t_pickup),
                 t_deliver=pd.Series(t_deliver), as_s=as_s)
    return orders, chain


# ------------------------------------------------------------------ picking + delivery
def make_ops(S, orders, C):
    pool = np.flatnonzero(C["delivered"])
    idx = np.sort(rng.choice(pool, N_OPS, replace=False))

    pick_start, pick_end = C["t_pick_start"].iloc[idx], C["t_pick_end"].iloc[idx]
    picker_global = C["store_idx"][idx] * PICKERS_PER_STORE + C["picker_j"][idx] + 1
    rider_global = C["store_idx"][idx] * RIDERS_PER_STORE + C["rider_j"][idx] + 1
    miss = C["missing"][idx]

    picking = pd.DataFrame({
        "order_id": orders.order_id.values[idx],
        "picker_id": [f"PICK{i:05d}" for i in picker_global],
        "pick_start": fmt(pick_start).values,
        "pick_end": fmt(pick_end).values,
        "items_picked": np.clip(C["items"][idx] - miss, 0, None),
        "items_missing": miss,
        "pick_duration_minutes": ((pick_end - pick_start).dt.total_seconds() / 60).round(1).values,
    })

    delivery = pd.DataFrame({
        "order_id": orders.order_id.values[idx],
        "rider_id": [f"RID{i:05d}" for i in rider_global],
        "assignment_time": fmt(C["t_assign"].iloc[idx]).values,
        "pickup_time": fmt(C["t_pickup"].iloc[idx]).values,
        "delivery_time": fmt(C["t_deliver"].iloc[idx]).values,
        "distance_km": C["dist"][idx],
        "assignment_delay_minutes": np.round(C["as_s"][idx] / 60, 1),
    })
    return picking, delivery


# ------------------------------------------------------------------ inventory
def make_inventory(S):
    rows = []
    for s in range(N_STORES):
        skus = rng.choice(SKU_UNIVERSE, SKU_PER_STORE, replace=False) + 1
        system = np.maximum(1, np.round(rng.gamma(3, 12, SKU_PER_STORE)).astype(int))
        out = rng.random(SKU_PER_STORE) < S["stockout"][s]
        noise_sd = 1 + 0.08 * system * (1 + 2 * S["stockout"][s] * 5)
        physical = np.clip(system + np.round(rng.normal(0, noise_sd)).astype(int), 1, None)
        physical = np.where(out, 0, physical)                # phantom stock: system>0, shelf=0
        var = physical - system
        acc = np.where(system > 0, np.minimum(physical, system) / system * 100, 100).round(1)
        rows.append(pd.DataFrame({
            "store_id": S["ids"][s],
            "sku_id": [f"SKU{k:04d}" for k in skus],
            "system_stock": system,
            "physical_stock": physical,
            "stockout": np.where(physical == 0, "Yes", "No"),
            "stock_variance": var,
            "inventory_accuracy_pct": acc,
        }))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


# ------------------------------------------------------------------ workforce
def make_workforce(S):
    n = N_WORKFORCE
    roles = rng.choice(["Picker", "Packer", "Store Associate", "Rider Coordinator", "Supervisor"],
                       n, p=[0.40, 0.20, 0.20, 0.10, 0.10])
    shift = rng.choice(["Morning", "Afternoon", "Evening", "Night"], n, p=[0.30, 0.25, 0.30, 0.15])
    stress = S["stress"][S["emp_store"]]
    completed = np.clip(np.round(rng.normal(45, 7, n)), 15, None).astype(int)
    pending = rng.poisson(5 * stress ** 1.5)
    ids = rng.permutation(n) + 1
    return pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in ids],
        "role": roles,
        "store_id": S["ids"][S["emp_store"]],
        "shift": shift,
        "tasks_completed": completed,
        "tasks_pending": pending,
    })


# ------------------------------------------------------------------ main
def main():
    v1 = ROOT / "data" / "raw_v1"
    if not v1.exists() and (RAW / "orders.csv").exists():
        v1.mkdir(parents=True)
        for f in RAW.glob("*"):
            shutil.copy2(f, v1 / f.name)
        print("v1 backed up ->", v1)

    S = make_stores()
    orders, C = make_orders_and_chain(S)
    picking, delivery = make_ops(S, orders, C)
    inventory = make_inventory(S)
    workforce = make_workforce(S)

    for name, df in dict(orders=orders, picking=picking, delivery=delivery,
                         inventory=inventory, workforce=workforce).items():
        df.to_csv(RAW / f"{name}.csv", index=False)
        print(f"{name:10s} {df.shape}")

    d = orders[orders.status.isin(["Delivered_On_Time", "Delivered_Late"])]
    print("\n--- sanity ---")
    print(orders.status.value_counts().to_string())
    print("breach rate (delivered):", round(d.sla_breached.mean(), 4))
    for t in (5, 10, 15):
        print(f"delay > {t} min:", round((d.delivery_delay_minutes > t).mean(), 4))
    sb = d.groupby("store_id").sla_breached.mean()
    print("store breach min/median/max:", sb.min().round(3), sb.median().round(3), sb.max().round(3))


if __name__ == "__main__":
    main()