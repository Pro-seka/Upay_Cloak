"""Behavioural anomaly detector: Isolation Forest over per-user deviation features.

Unsupervised (never sees labels). Score is a 0..1 percentile against the training
distribution, so "0.99" means "more unusual than 99% of training transactions".
"""
import numpy as np
from sklearn.ensemble import IsolationForest

ANOM_FEATURES = ["amount_z", "hour_freq", "is_new_device", "is_new_location", "is_new_recipient",
                 "txn_count_1h", "amount_to_balance", "log_secs_since_last", "device_users_count"]


class BehaviorAnomaly:
    def __init__(self, seed=42):
        self.model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        self.ref = None

    def fit(self, feats):
        X = feats[ANOM_FEATURES].values
        self.model.fit(X)
        self.ref = np.sort(-self.model.decision_function(X))
        return self

    def raw(self, feats):
        return -self.model.decision_function(feats[ANOM_FEATURES].values)

    def score(self, feats):
        """0..1 anomaly percentile vs. the training distribution."""
        return np.searchsorted(self.ref, self.raw(feats)) / len(self.ref)
