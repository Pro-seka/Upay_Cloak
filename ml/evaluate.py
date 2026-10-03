"""Metrics: ML quality + business impact (not just AUC)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.decision import ACTIONS, FRICTION, STOP_RATE, to_action_idx


def ranking_metrics(y, s, ks=(0.005, 0.01, 0.02)):
    m = dict(roc_auc=round(roc_auc_score(y, s), 4), pr_auc=round(average_precision_score(y, s), 4))
    order = np.argsort(-s)
    for k in ks:
        n = int(len(y) * k); top = y[order[:n]]
        m[f"precision@{k:.1%}"] = round(float(top.mean()), 4)
        m[f"recall@{k:.1%}"] = round(float(top.sum() / y.sum()), 4)
    return m


def business_impact(df, score, th):
    a = to_action_idx(score, th); y = df.is_fraud.values; amt = df.amount.values
    fraud_total = amt[y == 1].sum()
    prevented = (amt * STOP_RATE[a])[y == 1].sum()
    legit = y == 0
    return dict(
        thresholds=dict(zip(ACTIONS[1:], map(float, th))),
        fraud_value_total_bdt=float(fraud_total), fraud_value_prevented_bdt=float(prevented),
        pct_fraud_value_prevented=round(float(prevented / fraud_total), 4),
        legit_txns_with_friction_pct=round(float((a[legit] > 0).mean()), 4),
        alert_rate_pct=round(float((a > 0).mean()), 4),
        friction_cost_bdt=float(FRICTION[a][legit].sum()),
        net_benefit_bdt=float(prevented - FRICTION[a][legit].sum()),
        analyst_queue_size=int((a >= 2).sum()),
        actions={ACTIONS[i]: int((a == i).sum()) for i in range(4)},
    )


def scenario_recall(df, score, th):
    """Share of each injected scenario that triggers any action (uses true scenario, ignores label noise)."""
    flagged = to_action_idx(score, th) > 0
    inj = df[df.scenario != "normal"]
    return {s: round(float(flagged[g.index].mean()), 3) for s, g in inj.groupby("scenario")}
