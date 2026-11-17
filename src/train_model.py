"""
MacroOps AI - train SLA risk models (XGBoost) for 3 stages + SHAP.

Run from repo root:  python -m src.train_model

Outputs
  models/sla_stage{1,2,3}.json   trained models
  models/meta.json               features, thresholds, metrics
  outputs/model/*.png            ROC / PR / SHAP charts
  outputs/model/metrics.csv      metrics table
  data/processed/risk_scores.parquet   scores for all modelled orders (for dashboard)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import OUTPUTS, PROCESSED, ROOT
from src.predict import shap_values
from src.features import (LABELS, STAGE_NAMES, STAGES, TARGET, load_model_data, time_split)

MODEL_DIR = ROOT / "models"
FIG = OUTPUTS / "model"
MODEL_DIR.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGET_RECALL = 0.70      # ops wants to catch at least 70% of breaches
SEED = 42

# shallow, regularised trees. No early stopping: the validation week has rain + promo
# (breach 33.7% vs 25.8% in train), so early stopping on it stops far too soon.
PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=20,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
              eval_metric="auc", random_state=SEED, n_jobs=-1)


def pick_threshold(y, p, target_recall=TARGET_RECALL):
    """Highest threshold whose recall >= target (max precision at that recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "flagged_pct": pred.mean() * 100,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    d = load_model_data()
    train, valid, test = time_split(d)
    print(f"rows: train {len(train)} | valid {len(valid)} | test {len(test)}")
    print(f"breach rate: train {train[TARGET].mean():.3f} | valid {valid[TARGET].mean():.3f} "
          f"| test {test[TARGET].mean():.3f}\n")

    meta, rows, scores = {"stages": {}}, [], d[["order_id", "order_time"]].copy()
    scores["split"] = np.where(d.order_time < "2026-09-17", "train",
                       np.where(d.order_time < "2026-09-24", "valid", "test"))
    curves, shap_store = {}, {}

    for stage, feats in STAGES.items():
        Xtr, ytr = train[feats], train[TARGET]
        Xva, yva = valid[feats], valid[TARGET]
        Xte, yte = test[feats], test[TARGET]

        # baseline: logistic regression
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr.fit(Xtr, ytr)
        lr_auc = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])

        model = XGBClassifier(**PARAMS)
        model.fit(Xtr, ytr)

        p_va = model.predict_proba(Xva)[:, 1]
        thr = pick_threshold(yva, p_va)
        p_te = model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, p_te, thr)
        m.update(stage=stage, name=STAGE_NAMES[stage], baseline_lr_auc=lr_auc,
                 threshold=thr, trees=PARAMS['n_estimators'])
        rows.append(m)

        model.save_model(MODEL_DIR / f"sla_stage{stage}.json")
        meta["stages"][str(stage)] = {"name": STAGE_NAMES[stage], "features": feats,
                                      "threshold": thr, "metrics": {k: (float(v) if not isinstance(v, str) else v)
                                                                     for k, v in m.items()}}
        scores[f"p_stage{stage}"] = model.predict_proba(d[feats])[:, 1]
        curves[stage] = (yte.values, p_te)

        # SHAP on test sample
        samp = Xte.sample(min(2000, len(Xte)), random_state=SEED)
        sv = shap_values(model, samp)
        shap_store[stage] = (sv, samp)

        print(f"stage {stage} ({STAGE_NAMES[stage]}): AUC {m['roc_auc']:.3f} (LR baseline {lr_auc:.3f}) | "
              f"PR-AUC {m['pr_auc']:.3f} | thr {thr:.3f} | precision {m['precision']:.2f} "
              f"recall {m['recall']:.2f} | flagged {m['flagged_pct']:.1f}% | trees {m['trees']}")

    # ---- metrics table
    mt = pd.DataFrame(rows).round(4)
    mt.to_csv(FIG / "metrics.csv", index=False)
    meta["target_recall"] = TARGET_RECALL
    meta["split"] = {"train": "2026-09-01..16", "valid": "2026-09-17..23", "test": "2026-09-24..30"}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    scores.to_parquet(PROCESSED / "risk_scores.parquet", index=False)

    # ---- ROC + PR curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for stage, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{STAGE_NAMES[stage]} (AUC {roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{STAGE_NAMES[stage]} (AP {average_precision_score(y, p):.2f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (test week)", xlabel="false positive rate", ylabel="true positive rate")
    ax[1].set(title="Precision-Recall (test week)", xlabel="recall", ylabel="precision")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "roc_pr.png", dpi=120); plt.close()

    # ---- SHAP charts
    for stage in (1, 3):
        sv, samp = shap_store[stage]
        names = [LABELS.get(c, c) for c in samp.columns]
        shap.summary_plot(sv, samp, feature_names=names, show=False, max_display=14)
        plt.title(f"SHAP summary - stage {stage}: {STAGE_NAMES[stage]}")
        plt.tight_layout(); plt.savefig(FIG / f"shap_summary_stage{stage}.png", dpi=120); plt.close()
        shap.summary_plot(sv, samp, feature_names=names, plot_type="bar", show=False, max_display=14)
        plt.title(f"Mean |SHAP| - stage {stage}")
        plt.tight_layout(); plt.savefig(FIG / f"shap_bar_stage{stage}.png", dpi=120); plt.close()

    sv, samp = shap_store[3]
    imp = pd.Series(np.abs(sv).mean(axis=0), index=samp.columns).sort_values(ascending=False)
    print("\nTop drivers (stage 3, mean |SHAP|):")
    print(imp.head(8).round(3).to_string())
    print("\nSaved: models/, outputs/model/, data/processed/risk_scores.parquet")


if __name__ == "__main__":
    main()