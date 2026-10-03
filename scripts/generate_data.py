"""Generate the UpayShield sample dataset (synthetic, Bangladesh-flavoured mobile money).

Writes a time-based split so nothing from the future leaks into training:
    data/train.csv  -> first 70% of the timeline
    data/test.csv   -> last 30% of the timeline

Normal behaviour: per-user habits (home city, device, usual hours, amount scale,
contacts, agents). Fraud scenarios (injected after day 5 so users have history):
    ato, scam_victim, mule_passthrough, structuring, rogue_agent
Realism knobs: salary-day spikes, legit large one-offs (rent, tuition), legit
travel / new phones, and ~6% unreported fraud (label noise).

Usage: python scripts/generate_data.py [--users 800] [--days 30] [--seed 42]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ["Dhaka", "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Barishal", "Rangpur", "Cumilla"]
CITY_P = [.45, .15, .08, .08, .08, .05, .05, .06]
TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT"]
START = pd.Timestamp("2026-01-01")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_FRAC = 0.70


def generate(n_users=800, n_agents=40, days=30, seed=42):
    rng = np.random.default_rng(seed)
    users = [f"U{i:05d}" for i in range(n_users)]
    agents = [f"A{i:03d}" for i in range(n_agents)]
    merchants = [f"M{i:03d}" for i in range(30)]
    prof = {}
    for i, u in enumerate(users):
        prof[u] = dict(
            city=str(rng.choice(CITIES, p=CITY_P)), device=f"D{i:05d}",
            hour_mu=rng.normal(14, 3), mu=rng.normal(7.0, 0.7),
            balance=float(np.exp(rng.normal(9.5, 0.8))),
            contacts=[str(c) for c in rng.choice(users, size=int(rng.integers(4, 12)), replace=False)],
            agents=[str(a) for a in rng.choice(agents, size=3, replace=False)],
            merchants=[str(m) for m in rng.choice(merchants, size=4, replace=False)],
            tp=rng.dirichlet([2, 3, 4, 3]),
        )
    rows = []

    def ts(day, hour):
        return START + pd.Timedelta(days=int(day), hours=int(hour) % 24,
                                    minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))

    def add(t, user, typ, amount, recip, agent, device, loc, bal, fraud=0, scen="normal"):
        rows.append(dict(ts=t, user_id=user, type=typ, amount=round(float(amount), 0),
                         recipient_id=recip, agent_id=agent, device_id=device, location=loc,
                         balance_before=round(float(bal), 0), is_fraud=fraud, scenario=scen))

    # ---------- normal behaviour ----------
    day_w = np.array([1 + 0.6 * (d in (0, 1, days - 4, days - 3, days - 2, days - 1)) for d in range(days)])
    day_w /= day_w.sum()
    for u in users:
        p = prof[u]
        for _ in range(rng.poisson(days * 1.3)):
            d = rng.choice(days, p=day_w)
            h = int(rng.normal(p["hour_mu"], 2.5)) % 24
            typ = str(rng.choice(TYPES, p=p["tp"]))
            amt = np.exp(rng.normal(p["mu"], 0.6)) * {"CASH_IN": 1.5, "CASH_OUT": 1.5, "TRANSFER": 1, "PAYMENT": .6}[typ]
            bal = p["balance"] * rng.uniform(.6, 1.4)
            big = rng.random() < 0.015                       # legit large one-off
            if big:
                amt *= rng.uniform(4, 10)
            amt = min(amt, 0.95 * bal)
            agent = None
            if typ == "TRANSFER":
                rec = str(rng.choice(p["contacts"])) if (rng.random() < 0.88 and not big) else str(rng.choice(users))
                if rec == u:
                    rec = p["contacts"][0]
            elif typ == "PAYMENT":
                rec = str(rng.choice(p["merchants"]))
            else:
                agent = str(rng.choice(p["agents"]))
                rec = agent
            dev = p["device"] if rng.random() > 0.02 else f"DNEW{rng.integers(1_000_000)}"   # new phone
            loc = p["city"] if rng.random() > 0.04 else str(rng.choice(CITIES))              # travel
            add(ts(d, h), u, typ, amt, rec, agent, dev, loc, bal)

    # ---------- fraud: mule rings ----------
    rings = [dict(mules=[f"MU{r}-{j}" for j in range(5)], device=f"DR{r}",
                  agents=[str(a) for a in rng.choice(agents, 2, replace=False)]) for r in range(6)]

    def launder(t0, ring, amount):
        hub, m2, m3 = ring["mules"][0], str(rng.choice(ring["mules"][1:3])), ring["mules"][3]
        t1 = t0 + pd.Timedelta(minutes=int(rng.integers(2, 10)))
        a1 = amount * rng.uniform(.85, .95)
        add(t1, hub, "TRANSFER", a1, m2, None, ring["device"], "Dhaka", amount, 1, "mule_passthrough")
        t2 = t1 + pd.Timedelta(minutes=int(rng.integers(3, 15)))
        a2 = a1 * rng.uniform(.9, .98)
        add(t2, m2, "TRANSFER", a2, m3, None, ring["device"], "Dhaka", a1, 1, "mule_passthrough")
        t3 = t2 + pd.Timedelta(minutes=int(rng.integers(3, 15)))
        ag = str(rng.choice(ring["agents"]))
        add(t3, m3, "CASH_OUT", a2 * .95, ag, ag, ring["device"], "Dhaka", a2, 1, "mule_passthrough")

    lo, hi = 5, days - 1
    for _ in range(40):                                       # account takeover
        u = str(rng.choice(users)); p = prof[u]; ring = rings[int(rng.integers(len(rings)))]
        t = ts(rng.integers(lo, hi), rng.integers(0, 6))
        dev = f"DX{rng.integers(1_000_000)}"
        loc = str(rng.choice([c for c in CITIES if c != p["city"]]))
        for _ in range(int(rng.integers(3, 6))):
            t += pd.Timedelta(minutes=int(rng.integers(1, 5)))
            bal = p["balance"] * rng.uniform(.8, 1.3)
            amt = bal * rng.uniform(.25, .6)
            add(t, u, "TRANSFER", amt, ring["mules"][0], None, dev, loc, bal, 1, "ato")
            if rng.random() < .6:
                launder(t, ring, amt)
    for _ in range(100):                                      # scam victims -> mule hub
        u = str(rng.choice(users)); p = prof[u]; ring = rings[int(rng.integers(len(rings)))]
        bal = p["balance"] * rng.uniform(.8, 1.4)
        amt = min(np.exp(p["mu"]) * rng.uniform(6, 20), .9 * bal)
        t = ts(rng.integers(lo, hi), int(rng.normal(p["hour_mu"], 3)) % 24)
        add(t, u, "TRANSFER", amt, ring["mules"][0], None, p["device"], p["city"], bal, 1, "scam_victim")
        if rng.random() < .8:
            launder(t, ring, amt)
    for _ in range(12):                                       # structuring under 50,000 BDT
        u = str(rng.choice(users)); p = prof[u]; ag = str(rng.choice(agents))
        t = ts(rng.integers(lo, hi), rng.integers(9, 14))
        for _ in range(int(rng.integers(5, 9))):
            t += pd.Timedelta(minutes=int(rng.integers(10, 40)))
            add(t, u, "CASH_OUT", rng.uniform(46000, 49900), ag, ag, p["device"], p["city"],
                rng.uniform(60000, 150000), 1, "structuring")
    for ag in rng.choice(agents, 2, replace=False):           # rogue agents: night bursts
        for day in rng.choice(np.arange(lo, hi), 6, replace=False):
            t = ts(day, rng.integers(22, 24))
            for _ in range(10):
                u = str(rng.choice(users)); p = prof[u]
                t += pd.Timedelta(minutes=int(rng.integers(1, 6)))
                add(t, u, "CASH_OUT", rng.uniform(8000, 30000), str(ag), str(ag), p["device"], p["city"],
                    rng.uniform(40000, 90000), 1, "rogue_agent")

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    f = df.index[df.is_fraud == 1]
    df.loc[rng.choice(f, int(.06 * len(f)), replace=False), "is_fraud"] = 0   # unreported fraud
    df.insert(0, "txn_id", [f"TXN-{i:07d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.users, days=a.days, seed=a.seed)
    cut = int(len(df) * TRAIN_FRAC)                           # df is time-sorted -> time-based split
    DATA_DIR.mkdir(exist_ok=True)
    df.iloc[:cut].to_csv(DATA_DIR / "train.csv", index=False)
    df.iloc[cut:].to_csv(DATA_DIR / "test.csv", index=False)
    for name, d in (("train", df.iloc[:cut]), ("test", df.iloc[cut:])):
        print(f"{name}: {len(d):,} txns | {d.ts.min():%Y-%m-%d} -> {d.ts.max():%Y-%m-%d} | fraud {d.is_fraud.mean():.2%}")
    print(df.scenario.value_counts().to_string())
