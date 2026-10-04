"""UpayShield ML v1: compare basic algorithms, pick the best, tune actions, report honestly.

Usage (from repo root):  python -m ml.train
Reads  data/transactions.csv   (one dataset; ml/data.py makes the time-based 70/30 train/test split)
Writes models/risk_engine.joblib, data/cache/scored_cache.parquet, reports/metrics_v1.json,
reports/model_comparison_v1.md, reports/shap_importance_v1.json, reports/*.png
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.anomaly import BehaviorAnomaly  # noqa: E402
from ml.decision import ACTIONS, FRICTION, STOP_RATE, action_idx, tune_thresholds  # noqa: E402
from ml.explain import Explainer  # noqa: E402
from ml.data import load_transactions  # noqa: E402
from ml.features import FEATURES, build_features  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
VERSION = "v1"            # names the report files (metrics_v1.json ...)
ARTIFACT_VERSION = "v1.1"  # stored inside models/risk_engine.joblib and returned by the API
FEEDBACK_LABELS = ROOT / "data" / "feedback_labels.csv"   # optional, written by ml/feedback.py


def ranking_metrics(y, s, ks=(0.01, 0.03)):
    m = dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))
    order = np.argsort(-s)
    for k in ks:
        top = y[order[: int(len(y) * k)]]
        m[f"precision@{k:.0%}"] = float(top.mean())
        m[f"recall@{k:.0%}"] = float(top.sum() / y.sum())
    return {k: round(float(v), 4) for k, v in m.items()}


def business_impact(amount, y, score, th):
    a = action_idx(score, th)
    legit = y == 0
    fraud_total = amount[y == 1].sum()
    prevented = (amount * STOP_RATE[a])[y == 1].sum()
    friction = FRICTION[a][legit].sum()
    return dict(
        thresholds=dict(zip(ACTIONS[1:], th)),
        fraud_value_total_bdt=float(fraud_total),
        fraud_value_prevented_bdt=float(prevented),
        pct_fraud_value_prevented=round(float(prevented / fraud_total), 4),
        legit_txns_with_friction_pct=round(float((a[legit] > 0).mean()), 4),
        alert_rate_pct=round(float((a > 0).mean()), 4),
        friction_cost_bdt=float(friction),
        net_benefit_bdt=float(prevented - friction),
        actions={ACTIONS[i]: int((a == i).sum()) for i in range(4)},
    )


def main():
    raw = load_transactions()
    # Features are built over the whole timeline in order (point-in-time), then split back by the `split` column.
    feats = build_features(raw)
    tr_all, te = (feats[feats.split == p].reset_index(drop=True) for p in ("train", "test"))

    # Analyst feedback (confirmed / dismissed cases) overrides TRAINING labels only; test labels stay untouched.
    if FEEDBACK_LABELS.exists():
        fb = pd.read_csv(FEEDBACK_LABELS).drop_duplicates("txn_id", keep="last").set_index("txn_id").is_fraud
        hit = tr_all.txn_id.isin(fb.index)
        tr_all.loc[hit, "is_fraud"] = tr_all.loc[hit, "txn_id"].map(fb).astype(int)
        print(f"applied {int(hit.sum())} analyst-feedback labels to the training data")

    # carve a validation slice (last 20% of train, by time) for model selection + threshold tuning
    cut = tr_all.ts.quantile(0.8)
    tr, va = tr_all[tr_all.ts < cut].reset_index(drop=True), tr_all[tr_all.ts >= cut].reset_index(drop=True)
    print(f"train {len(tr):,} | valid {len(va):,} | test {len(te):,} | fraud rate "
          f"{tr.is_fraud.mean():.2%} / {va.is_fraud.mean():.2%} / {te.is_fraud.mean():.2%}")

    Xtr, ytr, Xva, yva, Xte, yte = tr[FEATURES], tr.is_fraud.values, va[FEATURES], va.is_fraud.values, te[FEATURES], te.is_fraud.values
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=min(pos_w, 10),
                                       random_state=SEED, verbose=-1),
    }
    results, val_scores, test_scores = {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_scores[name], test_scores[name] = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        results[name] = dict(valid=ranking_metrics(yva, val_scores[name]), test=ranking_metrics(yte, test_scores[name]))

    # unsupervised baseline: Isolation Forest on behaviour-deviation features only (never sees labels)
    anomaly = BehaviorAnomaly(SEED).fit(tr)
    results["Isolation Forest (unsupervised)"] = dict(
        valid=ranking_metrics(yva, anomaly.score(va)), test=ranking_metrics(yte, anomaly.score(te)))

    # pick the best supervised model by VALIDATION PR-AUC (test is only used for the final report)
    best = max(models, key=lambda n: results[n]["valid"]["pr_auc"])
    print(f"best model on validation PR-AUC: {best}")
    th = tune_thresholds(val_scores[best], va.amount.values, yva)
    impact = business_impact(te.amount.values, yte, test_scores[best], th)

    flagged = action_idx(test_scores[best], th) > 0
    scen = te[te.scenario != "normal"]
    scenario_recall = {s: round(float(flagged[g.index].mean()), 3) for s, g in scen.groupby("scenario")}

    # ---------- save artefacts ----------
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(dict(model=models[best], anomaly=anomaly, name=best, features=FEATURES, thresholds=th,
                     version=ARTIFACT_VERSION), ROOT / "models" / "risk_engine.joblib")

    # global explainability: mean |contribution| per feature on a test sample
    shap_imp = Explainer(models[best], FEATURES).global_importance(Xte.sample(min(1500, len(Xte)), random_state=1))
    (ROOT / "reports" / f"shap_importance_{VERSION}.json").write_text(json.dumps(shap_imp.round(4).to_dict(), indent=2))
    metrics = dict(version=VERSION, best_model=best, split=dict(train=len(tr), valid=len(va), test=len(te)),
                   models=results, business_impact=impact, scenario_recall=scenario_recall,
                   notes="Synthetic data with 6% label noise. Absolute numbers are optimistic; use for model comparison.")
    (ROOT / "reports" / f"metrics_{VERSION}.json").write_text(json.dumps(metrics, indent=2))

    lines = ["| Model | ROC-AUC | PR-AUC | Precision@3% | Recall@3% |", "|---|---|---|---|---|"]
    for n, r in results.items():
        t = r["test"]
        lines.append(f"| {n} | {t['roc_auc']} | {t['pr_auc']} | {t['precision@3%']} | {t['recall@3%']} |")
    (ROOT / "reports" / f"model_comparison_{VERSION}.md").write_text("\n".join(lines) + "\n")

    # plots: precision-recall curves + feature importance of the best model
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in models:
        p, r, _ = precision_recall_curve(yte, test_scores[n])
        ax[0].plot(r, p, label=f"{n} (AP={results[n]['test']['pr_auc']:.2f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall on test set")
    ax[0].legend(loc="lower left")
    bm = models[best]
    imp = getattr(bm, "feature_importances_", None)
    if imp is None:
        imp = np.abs(bm[-1].coef_[0])
    s = pd.Series(imp, index=FEATURES).sort_values().tail(12)
    ax[1].barh(s.index, s.values)
    ax[1].set(title=f"Top features ({best})")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / f"model_comparison_{VERSION}.png", dpi=130)

    print("\n" + "\n".join(lines))
    print(f"\nThresholds (otp, hold, block): {th}")
    print(f"Fraud value prevented: {impact['pct_fraud_value_prevented']:.1%} | legit txns with friction: "
          f"{impact['legit_txns_with_friction_pct']:.2%} | net benefit BDT {impact['net_benefit_bdt']:,.0f}")
    print("Recall by scenario (any action triggered):", scenario_recall)
    print("Top global factors:", ", ".join(f"{k} ({v:.3f})" for k, v in shap_imp.head(5).items()))

    from ml.cache import build_cache
    build_cache()          # so the backend boots instantly from data/cache/scored_cache.parquet


if __name__ == "__main__":
    main()
