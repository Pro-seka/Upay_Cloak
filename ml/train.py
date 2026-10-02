"""Train + evaluate the full ML stack.  Usage: python -m ml.train"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.anomaly import BehaviorAnomaly
from ml.decision import tune_thresholds
from ml.evaluate import business_impact, ranking_metrics, scenario_recall
from ml.explain import Explainer
from ml.features import FEATURES, build_features
from ml.score import W_ANOM, W_MODEL, RiskEngine


def main():
    csv = ROOT / "data" / "transactions.csv"
    if not csv.exists():
        from scripts.generate_data import generate
        csv.parent.mkdir(exist_ok=True); generate().to_csv(csv, index=False)
    df = pd.read_csv(csv, parse_dates=["ts"])
    print(f"building features for {len(df):,} txns ..."); feats = build_features(df)

    # time-based split (no shuffling -> no future leakage): 60% train / 15% valid / 25% test
    q = feats.ts.quantile([.6, .75]).values
    tr, va, te = feats[feats.ts < q[0]], feats[(feats.ts >= q[0]) & (feats.ts < q[1])], feats[feats.ts >= q[1]]
    tr, va, te = (d.reset_index(drop=True) for d in (tr, va, te))
    print(f"train {len(tr):,} | valid {len(va):,} | test {len(te):,} | fraud rate train {tr.is_fraud.mean():.2%}")

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(tr[FEATURES], tr.is_fraud, eval_set=[(va[FEATURES], va.is_fraud)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(30, verbose=False)])
    anomaly = BehaviorAnomaly().fit(tr)

    def blend(d):
        p, a = model.predict_proba(d[FEATURES])[:, 1], anomaly.score(d)
        return W_MODEL * p + W_ANOM * a, p, a

    sv, _, _ = blend(va)
    th = tune_thresholds(sv, va.amount.values, va.is_fraud.values)
    print("tuned thresholds (otp, hold, block):", th)

    s, p, a = blend(te); y = te.is_fraud.values
    metrics = dict(
        data=dict(n_test=len(te), test_fraud_rate=round(float(y.mean()), 4), best_iteration=int(model.best_iteration_)),
        lightgbm_only=ranking_metrics(y, p), isolation_forest_only=ranking_metrics(y, a), blended=ranking_metrics(y, s),
        business_impact=business_impact(te, s, th), scenario_recall=scenario_recall(te, s, th),
        notes="Synthetic data with 6% label noise and legit large one-offs; treat absolute numbers as indicative only.",
    )
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(dict(model=model, anomaly=anomaly, features=FEATURES, thresholds=th), ROOT / "models" / "risk_engine.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # global SHAP importance + sample cases (one per scenario, highest risk)
    imp = pd.Series(abs(Explainer(model, FEATURES).shap_values(te[FEATURES].sample(2000, random_state=1))).mean(0),
                    index=FEATURES).sort_values(ascending=False)
    (ROOT / "reports" / "shap_importance.json").write_text(json.dumps(imp.round(4).to_dict(), indent=2))
    te["risk"] = s
    pick = te[te.scenario != "normal"].sort_values("risk", ascending=False).groupby("scenario").head(1)
    pick = pd.concat([pick, te[te.scenario == "normal"].sort_values("risk", ascending=False).head(1)])
    cases = RiskEngine(ROOT / "models" / "risk_engine.joblib").score_frame(pick)
    (ROOT / "reports" / "sample_cases.json").write_text(json.dumps(cases, indent=2, default=str))

    print(json.dumps({k: metrics[k] for k in ("lightgbm_only", "isolation_forest_only", "blended", "scenario_recall")}, indent=1))
    bi = metrics["business_impact"]
    print(f"\nBusiness impact: prevented {bi['pct_fraud_value_prevented']:.1%} of fraud value | "
          f"{bi['legit_txns_with_friction_pct']:.2%} legit txns got friction | net benefit BDT {bi['net_benefit_bdt']:,.0f}")
    print("Top SHAP features:", ", ".join(imp.index[:6]))


if __name__ == "__main__":
    main()
