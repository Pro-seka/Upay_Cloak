"""RiskEngine: the contract Person B's FastAPI `/score` endpoint can call.

Output per transaction answers the three judging questions:
  what_happened -> facts | why_risky -> scores, SHAP reasons, rule trace, tags | what_next -> action
"""
import joblib
import numpy as np
import pandas as pd

from ml.decision import recommend
from ml.explain import Explainer
from ml.rules import rule_trace

W_MODEL, W_ANOM = 0.8, 0.2


class RiskEngine:
    def __init__(self, path="models/risk_engine.joblib"):
        art = joblib.load(path)
        self.model, self.anomaly = art["model"], art["anomaly"]
        self.features, self.th = art["features"], art["thresholds"]
        self.explainer = Explainer(self.model, self.features)

    def blend(self, feats: pd.DataFrame):
        p = self.model.predict_proba(feats[self.features])[:, 1]
        a = self.anomaly.score(feats)
        return W_MODEL * p + W_ANOM * a, p, a

    def score_frame(self, feats: pd.DataFrame, explain_limit=None):
        risk, p, a = self.blend(feats)
        feats = feats.reset_index(drop=True)
        flagged = [i for i in range(len(feats)) if recommend(risk[i], self.th) != "allow"][:explain_limit]
        reasons = dict(zip(flagged, self.explainer.reasons(feats.loc[flagged, self.features]))) if flagged else {}
        cases = []
        for i, r in feats.iterrows():
            trace = rule_trace(r)
            cases.append(dict(
                txn_id=r.txn_id, user_id=r.user_id,
                what_happened=dict(type=r.type, amount_bdt=r.amount, recipient=r.recipient_id, device=r.device_id,
                                   location=r.location, time=str(r.ts)),
                why_risky=dict(risk_score=round(float(risk[i]), 4), model_score=round(float(p[i]), 4),
                               anomaly_score=round(float(a[i]), 4), reasons=reasons.get(i, []),
                               rule_trace=trace, tags=sorted({h["tag"] for h in trace})),
                what_next=dict(action=recommend(risk[i], self.th)),
            ))
        return cases
