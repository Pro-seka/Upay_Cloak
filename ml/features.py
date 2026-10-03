"""Leak-free, point-in-time feature pipeline.

Transactions are replayed in time order. Every feature is computed from state
*before* the current transaction, then the state is updated. So a test-set row
only ever "knows" the past, which is exactly how it works in production.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT"]
THRESHOLD = 50_000

FEATURES = [
    "amount", "log_amount", "type_code", "hour", "is_night",
    # behavioural deviation (vs. this user's own history)
    # (user_txn_count is still computed in the frame but NOT used as a model input:
    #  mule accounts are brand new in the synthetic data, so it would be a data artifact)
    "amount_z", "hour_freq", "amount_to_balance", "log_secs_since_last",
    # account takeover signals
    "is_new_device", "is_new_location", "is_new_recipient", "txn_count_1h", "amount_sum_1h",
    # mule / network signals
    "recipient_unique_senders", "recipient_inbound_count", "recipient_age_txns",
    "device_users_count", "inbound_1h", "passthrough_ratio",
    # structuring / agent signals
    "near_thr_24h", "agent_txn_1h", "agent_cashout_ratio",
]


def _user_state():
    return dict(n=0, mean=0.0, m2=0.0, hours=np.zeros(24), devices=set(), locs=set(), recips=set(),
                recent=deque(), last=None, inbound=deque(), near=deque())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    U = defaultdict(_user_state)
    R_senders, R_in = defaultdict(set), defaultdict(int)
    D_users = defaultdict(set)
    A = defaultdict(lambda: dict(out=0, inn=0, recent=deque()))
    out = []
    for r in df.itertuples(index=False):
        t = r.ts.timestamp()
        h = r.ts.hour
        u = U[r.user_id]
        warm = u["n"] >= 5                      # need some history before "deviation" means anything
        la = np.log1p(r.amount)
        for dq, win in ((u["recent"], 3600), (u["inbound"], 3600), (u["near"], 86400)):
            while dq and dq[0][0] < t - win:
                dq.popleft()
        std = max(np.sqrt(u["m2"] / (u["n"] - 1)), 0.3) if u["n"] > 1 else 1.0
        near = THRESHOLD * .9 <= r.amount < THRESHOLD
        is_cash = r.type in ("CASH_IN", "CASH_OUT")
        a = A[r.agent_id] if is_cash and r.agent_id else None
        if a:
            while a["recent"] and a["recent"][0] < t - 3600:
                a["recent"].popleft()
        in1h = sum(x[1] for x in u["inbound"])
        rec_n = U[r.recipient_id]["n"] if r.type == "TRANSFER" else 100
        out.append(dict(
            log_amount=la, type_code=TYPES.index(r.type), hour=h, is_night=int(h < 6 or h >= 23),
            amount_z=(la - u["mean"]) / std if warm else 0.0,
            hour_freq=(u["hours"][h] + u["hours"][(h - 1) % 24] + u["hours"][(h + 1) % 24]) / u["n"] if warm else .5,
            amount_to_balance=r.amount / max(r.balance_before, 1),
            user_txn_count=u["n"],
            log_secs_since_last=np.log1p(t - u["last"]) if u["last"] else np.log1p(7 * 86400),
            is_new_device=int(warm and r.device_id not in u["devices"]),
            is_new_location=int(warm and r.location not in u["locs"]),
            is_new_recipient=int(warm and r.recipient_id not in u["recips"]),
            txn_count_1h=len(u["recent"]) + 1,
            amount_sum_1h=sum(x[1] for x in u["recent"]) + r.amount,
            recipient_unique_senders=len(R_senders[r.recipient_id]),
            recipient_inbound_count=R_in[r.recipient_id],
            recipient_age_txns=rec_n,
            device_users_count=len(D_users[r.device_id] | {r.user_id}),
            inbound_1h=in1h,
            passthrough_ratio=min(r.amount / in1h, 2.0) if in1h > 0 else 0.0,
            near_thr_24h=len(u["near"]) + int(near),
            agent_txn_1h=len(a["recent"]) + 1 if a else 0,
            agent_cashout_ratio=a["out"] / (a["out"] + a["inn"] + 1) if a else 0.0,
        ))
        # ---- update state AFTER computing features ----
        u["n"] += 1
        d = la - u["mean"]
        u["mean"] += d / u["n"]
        u["m2"] += d * (la - u["mean"])
        u["hours"][h] += 1
        u["devices"].add(r.device_id)
        u["locs"].add(r.location)
        u["recips"].add(r.recipient_id)
        u["recent"].append((t, r.amount))
        u["last"] = t
        if near:
            u["near"].append((t, r.amount))
        R_senders[r.recipient_id].add(r.user_id)
        R_in[r.recipient_id] += 1
        if r.type == "TRANSFER":
            U[r.recipient_id]["inbound"].append((t, r.amount))
        D_users[r.device_id].add(r.user_id)
        if a:
            a["recent"].append(t)
            a["out" if r.type == "CASH_OUT" else "inn"] += 1
    return pd.concat([df, pd.DataFrame(out)], axis=1)
