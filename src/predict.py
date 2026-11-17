"""
MacroOps AI - score one order and explain why.

    python -m src.predict                 # explains a high-risk order from the test week
    python -m src.predict ORD-012345      # explain a given order

Reused later by the dashboard, FastAPI and the AI Copilot tool `explain_order_risk`.
"""
import json
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from src.config import PROCESSED, ROOT
from src.features import LABELS, STAGE_NAMES, STAGES, TARGET, build_features

MODEL_DIR = ROOT / "models"


@lru_cache(maxsize=1)
def _meta() -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text())


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Exact SHAP values (log-odds) from XGBoost itself.
    Same numbers as the shap library's tree explainer, without shap/xgboost version clashes."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contrib[:, :-1]          # last column = bias


@lru_cache(maxsize=3)
def load_stage(stage: int):
    model = XGBClassifier()
    model.load_model(MODEL_DIR / f"sla_stage{stage}.json")
    return model


@lru_cache(maxsize=1)
def _orders() -> pd.DataFrame:
    """Feature table for all orders (store-load features need every order)."""
    m = pd.read_parquet(PROCESSED / "master.parquet")
    return build_features(m).set_index("order_id")


def get_threshold(stage: int) -> float:
    return _meta()["stages"][str(stage)]["threshold"]


def models_ready() -> bool:
    return (MODEL_DIR / "meta.json").exists() and (PROCESSED / "risk_scores.parquet").exists()


def risk_level(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "HIGH"
    if prob >= 0.6 * threshold:
        return "MEDIUM"
    return "LOW"


def available_stage(row: pd.Series) -> int:
    """Highest stage whose inputs exist for this order."""
    if pd.isna(row.get("distance_km")):
        return 0
    if pd.isna(row.get("pick_duration_minutes")):
        return 1
    if pd.isna(row.get("assignment_delay_minutes")):
        return 2
    return 3


def explain_order(order_id: str, stage: int | None = None, top_n: int = 5) -> dict:
    orders = _orders()
    if order_id not in orders.index:
        return {"error": f"order {order_id} not found"}
    row = orders.loc[order_id]
    if not bool(row.is_delivered):
        return {"error": f"order {order_id} was {row.status}; SLA risk applies to delivered orders"}

    max_stage = available_stage(row)
    if max_stage == 0:
        return {"error": f"order {order_id} has no picking/delivery detail (distance unknown); cannot score"}
    stage = min(stage or max_stage, max_stage)

    feats = STAGES[stage]
    X = row[feats].to_frame().T.astype(float)
    model = load_stage(stage)
    prob = float(model.predict_proba(X)[:, 1][0])
    thr = _meta()["stages"][str(stage)]["threshold"]
    sv = shap_values(model, X)[0]

    order_idx = np.argsort(-np.abs(sv))[:top_n]
    drivers = []
    for i in order_idx:
        f = feats[i]
        drivers.append({
            "feature": f,
            "label": LABELS.get(f, f),
            "value": round(float(X.iloc[0, i]), 2),
            "shap": round(float(sv[i]), 3),
            "effect": "raises risk" if sv[i] > 0 else "lowers risk",
        })

    level = risk_level(prob, thr)
    up = [d for d in drivers if d["shap"] > 0]
    text = (f"Order {order_id} ({row.store_id}): {level} SLA risk, {prob:.0%} chance of breach "
            f"(stage {stage}: {STAGE_NAMES[stage]}). ")
    if up:
        text += "Main reasons: " + "; ".join(f"{d['label']} = {d['value']:g}" for d in up[:3]) + "."
    else:
        text += "No strong risk drivers."

    out = {"order_id": order_id, "store_id": row.store_id, "stage": stage,
           "stage_name": STAGE_NAMES[stage], "probability": round(prob, 3),
           "threshold": round(thr, 3), "risk_level": level,
           "actual_breached": int(row[TARGET]) if not pd.isna(row[TARGET]) else None,
           "drivers": drivers, "summary": text}
    return out


def top_risk_orders(n: int = 10, split: str = "test") -> pd.DataFrame:
    """Highest stage-3 risk orders from saved scores (for dashboard / copilot)."""
    s = pd.read_parquet(PROCESSED / "risk_scores.parquet")
    s = s[s.split == split].sort_values("p_stage3", ascending=False).head(n)
    return s[["order_id", "order_time", "p_stage1", "p_stage2", "p_stage3"]].round(3)


if __name__ == "__main__":
    oid = sys.argv[1] if len(sys.argv) > 1 else top_risk_orders(1).order_id.iloc[0]
    for st in (1, 2, 3):
        r = explain_order(oid, stage=st)
        print(f"\n--- stage {st} ---")
        if "error" in r:
            print(r["error"])
            continue
        print(r["summary"])
        print("actual breached:", r["actual_breached"])
        for d in r["drivers"]:
            print(f"  {d['label']:34s} {d['value']:>8g}  shap {d['shap']:+.2f}  ({d['effect']})")