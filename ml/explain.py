"""SHAP explanations -> plain-language reasons."""
import numpy as np
import shap

TEXT = {
    "amount_z": lambda v: f"Amount is {v:.1f} std devs above this user's usual spend",
    "amount": lambda v: f"Large amount (BDT {v:,.0f})",
    "is_new_device": lambda v: "Device never used before by this user",
    "is_new_location": lambda v: "Unusual location for this user",
    "is_new_recipient": lambda v: "First-time recipient",
    "hour_freq": lambda v: "User rarely transacts at this hour",
    "is_night": lambda v: "Night-time transaction",
    "txn_count_1h": lambda v: f"{v:.0f} transactions in the last hour (velocity burst)",
    "amount_sum_1h": lambda v: f"BDT {v:,.0f} moved in the last hour",
    "amount_to_balance": lambda v: f"Moves {v:.0%} of the wallet balance",
    "recipient_unique_senders": lambda v: f"Recipient has received from {v:.0f} different senders",
    "recipient_inbound_count": lambda v: f"Recipient has {v:.0f} inbound transfers (fan-in)",
    "recipient_age_txns": lambda v: f"Recipient account has little history ({v:.0f} txns)",
    "device_users_count": lambda v: f"Device shared by {v:.0f} accounts",
    "inbound_1h": lambda v: f"Wallet received BDT {v:,.0f} within the last hour",
    "passthrough_ratio": lambda v: f"Forwarding {v:.0%} of money just received",
    "near_thr_24h": lambda v: f"{v:.0f} cash-outs just under the 50,000 BDT threshold in 24h",
    "agent_txn_1h": lambda v: f"Agent handled {v:.0f} transactions in the last hour",
    "agent_cashout_ratio": lambda v: f"Agent cash-out share is {v:.0%}",
}


class Explainer:
    def __init__(self, model, features):
        self.exp, self.features = shap.TreeExplainer(model), features

    def shap_values(self, X):
        sv = self.exp.shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        if sv.ndim == 3: sv = sv[:, :, 1]
        return sv

    def reasons(self, X, k=4):
        sv, out = self.shap_values(X), []
        for i in range(len(X)):
            order = np.argsort(-sv[i])[:k]
            out.append([dict(feature=self.features[j], value=float(X.iloc[i, j]), shap=round(float(sv[i, j]), 3),
                             text=TEXT.get(self.features[j], lambda v, f=self.features[j]: f)(float(X.iloc[i, j])))
                        for j in order if sv[i, j] > 0])
        return out
