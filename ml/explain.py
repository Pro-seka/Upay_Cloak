"""Local explanations: which factors pushed THIS transaction's risk up or down.

    Explainer(model, FEATURES).explain(X)  ->  one dict per row:
        base_value        average model output (what a "typical" transaction scores)
        risk              base_value + sum(contributions)  (== model output for that row)
        top_positive      factors that INCREASED risk, biggest first, e.g. amount_z +0.28
        top_negative      factors that REDUCED risk, biggest first, e.g. is_new_recipient -0.08
        unit              "probability" for Random Forest, "log-odds" for LightGBM / Logistic Regression

Tree models use SHAP (TreeExplainer). Logistic Regression uses coefficient x standardised value,
which is its exact additive decomposition. Contributions are computed on demand, for any row.
"""
import numpy as np
import pandas as pd
import shap

# feature -> (text when it INCREASES risk, text when it REDUCES risk); each takes the feature value
TEXT = {
    "amount_z": (lambda v: f"Amount is {v:.1f} std devs above this user's usual spend",
                 lambda v: "Amount is in line with this user's usual spend"),
    "is_new_device": (lambda v: "Device never used before by this user", lambda v: "Usual device for this user"),
    "is_new_location": (lambda v: "Unusual location for this user", lambda v: "Usual location for this user"),
    "is_new_recipient": (lambda v: "First-time recipient", lambda v: "Known recipient (paid before)"),
    "hour_freq": (lambda v: "User rarely transacts at this hour", lambda v: "Usual hour for this user"),
    "is_night": (lambda v: "Night-time transaction", lambda v: "Daytime transaction"),
    "txn_count_1h": (lambda v: f"{v:.0f} transactions in the last hour (velocity burst)",
                     lambda v: "Normal transaction pace"),
    "amount_sum_1h": (lambda v: f"BDT {v:,.0f} moved in the last hour", lambda v: "Little money moved recently"),
    "amount": (lambda v: f"Large amount (BDT {v:,.0f})", lambda v: f"Modest amount (BDT {v:,.0f})"),
    "log_amount": (lambda v: f"Large amount (BDT {np.expm1(v):,.0f})", lambda v: f"Modest amount (BDT {np.expm1(v):,.0f})"),
    "amount_to_balance": (lambda v: f"Moves {v:.0%} of the wallet balance", lambda v: f"Only {v:.0%} of the wallet balance"),
    "log_secs_since_last": (lambda v: "Unusually soon after the previous transaction", lambda v: "Normal gap since last transaction"),
    "recipient_unique_senders": (lambda v: f"Recipient has received from {v:.0f} different senders",
                                 lambda v: "Recipient has few distinct senders"),
    "recipient_inbound_count": (lambda v: f"Recipient has {v:.0f} inbound transfers (fan-in)",
                                lambda v: "Recipient has little inbound traffic"),
    "recipient_age_txns": (lambda v: f"Recipient account has little history ({v:.0f} txns)",
                           lambda v: "Recipient is an established account"),
    "device_users_count": (lambda v: f"Device shared by {v:.0f} accounts", lambda v: "Device used by a single account"),
    "inbound_1h": (lambda v: f"Wallet received BDT {v:,.0f} within the last hour", lambda v: "No recent incoming money"),
    "passthrough_ratio": (lambda v: f"Forwarding {v:.0%} of money just received", lambda v: "Not forwarding received money"),
    "near_thr_24h": (lambda v: f"{v:.0f} cash-outs just under the 50,000 BDT threshold in 24h",
                     lambda v: "No threshold-hugging cash-outs"),
    "agent_txn_1h": (lambda v: f"Agent handled {v:.0f} transactions in the last hour", lambda v: "Normal agent volume"),
    "agent_cashout_ratio": (lambda v: f"Agent cash-out share is {v:.0%}", lambda v: "Balanced agent cash-in/out"),
    "type_code": (lambda v: "Transaction type carries higher risk", lambda v: "Transaction type carries lower risk"),
    "hour": (lambda v: f"Unusual hour of day ({v:.0f}:00)", lambda v: f"Typical hour of day ({v:.0f}:00)"),
}


def describe(feature, value, contribution):
    up, down = TEXT.get(feature, (lambda v: feature, lambda v: feature))
    return (up if contribution > 0 else down)(float(value))


class Explainer:
    def __init__(self, model, features):
        self.model, self.features = model, list(features)
        self.is_linear = hasattr(model, "steps")                      # sklearn Pipeline(scaler, LogisticRegression)
        self.unit = "log-odds" if self.is_linear or "LGBM" in type(model).__name__ else "probability"
        self.tree = None if self.is_linear else shap.TreeExplainer(model)

    # ---- raw contributions: (base_value, matrix[n_rows, n_features]) ----
    def contributions(self, X: pd.DataFrame):
        X = X[self.features]
        if self.is_linear:
            scaler, lr = self.model[0], self.model[-1]
            return float(lr.intercept_[0]), lr.coef_[0] * scaler.transform(X)
        sv = self.tree.shap_values(X)
        base = np.atleast_1d(self.tree.expected_value)
        if isinstance(sv, list):                                      # older shap: list per class
            sv, base = sv[1], base[-1]
        elif sv.ndim == 3:                                            # newer shap: (rows, features, classes)
            sv, base = sv[:, :, 1], base[-1]
        else:
            base = base[0]
        return float(base), sv

    def explain(self, X: pd.DataFrame, k_pos=4, k_neg=3):
        base, sv = self.contributions(X)
        X = X[self.features]
        out = []
        for i in range(len(X)):
            order = np.argsort(-sv[i])
            def item(j):
                v = float(X.iloc[i, j])
                return dict(feature=self.features[j], value=round(v, 4), contribution=round(float(sv[i, j]), 4),
                            direction="increases_risk" if sv[i, j] > 0 else "reduces_risk",
                            text=describe(self.features[j], v, sv[i, j]))
            def top(js, k, keep):
                picked, seen = [], set()
                for j in js:
                    if not keep(sv[i, j]):
                        break
                    it = item(j)
                    if it["text"] not in seen:          # amount / log_amount etc. would read as duplicates
                        seen.add(it["text"])
                        picked.append(it)
                    if len(picked) == k:
                        break
                return picked
            pos = top(order, k_pos, lambda c: c > 1e-4)
            neg = top(order[::-1], k_neg, lambda c: c < -1e-4)
            out.append(dict(base_value=round(base, 4), risk=round(base + float(sv[i].sum()), 4),
                            unit=self.unit, top_positive=pos, top_negative=neg))
        return out

    def global_importance(self, X: pd.DataFrame):
        """Mean |contribution| per feature (for the dashboard / README)."""
        _, sv = self.contributions(X)
        return pd.Series(np.abs(sv).mean(0), index=self.features).sort_values(ascending=False)
