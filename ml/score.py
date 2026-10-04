"""Scoring entry point for the backend: what-happened / why-risky / what-next per transaction.

    python -m ml.score        # demo: the 3 riskiest test transactions as case JSON

Model artifact (models/risk_engine.joblib) holds: model, anomaly, features, thresholds (+ name, version).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.decision import recommend  # noqa: E402
from ml.explain import Explainer  # noqa: E402
from ml.rules import rule_trace  # noqa: E402,F401  (single shared copy; cache.py imports it from here)

MODEL_PATH = ROOT / "models" / "risk_engine.joblib"


class RiskEngine:
    def __init__(self, path=MODEL_PATH):
        art = joblib.load(path)
        self.model, self.anomaly = art["model"], art["anomaly"]
        self.features, self.th = art["features"], art["thresholds"]
        self.version, self.name = art.get("version", "?"), art.get("name", "?")
        self.explainer = Explainer(self.model, self.features)

    def predict(self, feats: pd.DataFrame):
        """Fast path (no explanations): risk, anomaly, action for every row."""
        risk = self.model.predict_proba(feats[self.features])[:, 1]
        anomaly = self.anomaly.score(feats)
        return risk, anomaly, [recommend(r, self.th) for r in risk]

    def score_frame(self, feats: pd.DataFrame, explain="flagged"):
        """feats must come from ml.features.build_features.
        explain: "flagged" (SHAP only for non-allow rows), "all", or "none"."""
        feats = feats.reset_index(drop=True)
        risk, anomaly, actions = self.predict(feats)
        idx = [i for i in range(len(feats)) if explain == "all" or (explain == "flagged" and actions[i] != "allow")]
        shap_out = dict(zip(idx, self.explainer.explain(feats.loc[idx]))) if idx else {}
        cases = []
        for i, r in feats.iterrows():
            trace = rule_trace(r)
            cases.append(dict(
                txn_id=r.txn_id, user_id=r.user_id,
                what_happened=dict(type=r.type, amount_bdt=r.amount, recipient=r.recipient_id,
                                   device=r.device_id, location=r.location, time=str(r.ts)),
                why_risky=dict(risk_score=round(float(risk[i]), 4), anomaly_score=round(float(anomaly[i]), 4),
                               feature_contributions=shap_out.get(i),
                               rule_trace=trace, tags=sorted({h["tag"] for h in trace})),
                what_next=dict(action=actions[i]),
                model_version=self.version,
            ))
        return cases


if __name__ == "__main__":
    from ml.cache import load_cache

    cache = load_cache()
    top = cache[cache.split == "test"].sort_values("risk_score", ascending=False).head(3)
    cases = RiskEngine().score_frame(top)
    print(json.dumps(cases, indent=2, default=str))
