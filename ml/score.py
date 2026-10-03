"""Scoring entry point for the backend: returns what-happened / why-risky / what-next per transaction.

    python -m ml.score            # demo: scores the 5 riskiest test transactions
v1 explanations = rule trace + tags + top deviating signals. SHAP reasons come in v2.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.decision import recommend  # noqa: E402
from ml.features import build_features  # noqa: E402


def rule_trace(r):
    hits = []

    def hit(rid, tag, text):
        hits.append(dict(rule_id=rid, tag=tag, text=text))

    if r["is_new_device"] and r["is_new_location"]:
        hit("ATO_01", "account_takeover", "New device AND new location for this wallet")
    if r["is_new_device"] and r["is_night"] and r["txn_count_1h"] >= 2:
        hit("ATO_02", "account_takeover", "New device at night with repeated transactions (velocity burst)")
    if r["type"] == "TRANSFER" and r["is_new_recipient"] and r["amount_z"] > 2.5:
        hit("SCAM_01", "scam_victim", "First-time recipient with an amount far above this user's norm")
    if r["inbound_1h"] > 0 and r["amount"] >= 0.7 * r["inbound_1h"] and r["type"] in ("TRANSFER", "CASH_OUT"):
        hit("MULE_01", "mule_passthrough", "Wallet just received money and is forwarding most of it")
    if r["device_users_count"] >= 3:
        hit("MULE_02", "mule_network", f"Device shared by {int(r['device_users_count'])} accounts")
    if r["near_thr_24h"] >= 2:
        hit("STRUCT_01", "structuring", "Repeated cash-outs just under the 50,000 BDT threshold")
    if r["agent_txn_1h"] >= 8 and (r["hour"] >= 22 or r["hour"] < 6):
        hit("AGENT_01", "agent_risk", "Agent processing an unusual late-night burst")
    return hits


class RiskEngine:
    def __init__(self, path=ROOT / "models" / "risk_model_v1.joblib"):
        art = joblib.load(path)
        self.model, self.features, self.th, self.version = art["model"], art["features"], art["thresholds"], art["version"]

    def score_frame(self, feats: pd.DataFrame):
        """feats must come from ml.features.build_features (point-in-time features)."""
        risk = self.model.predict_proba(feats[self.features])[:, 1]
        cases = []
        for i, r in feats.reset_index(drop=True).iterrows():
            trace = rule_trace(r)
            cases.append(dict(
                txn_id=r.txn_id, user_id=r.user_id,
                what_happened=dict(type=r.type, amount_bdt=r.amount, recipient=r.recipient_id,
                                   device=r.device_id, location=r.location, time=str(r.ts)),
                why_risky=dict(risk_score=round(float(risk[i]), 4), rule_trace=trace,
                               tags=sorted({h["tag"] for h in trace})),
                what_next=dict(action=recommend(risk[i], self.th)),
                model_version=self.version,
            ))
        return cases


if __name__ == "__main__":
    full = pd.concat([pd.read_csv(ROOT / "data" / f"{p}.csv", parse_dates=["ts"]) for p in ("train", "test")],
                     ignore_index=True)
    feats = build_features(full)
    test = feats.iloc[-len(pd.read_csv(ROOT / "data" / "test.csv")):].reset_index(drop=True)
    eng = RiskEngine()
    cases = eng.score_frame(test)
    cases.sort(key=lambda c: -c["why_risky"]["risk_score"])
    print(json.dumps(cases[:3], indent=2, default=str))
