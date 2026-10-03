"""Analyst feedback loop: confirmed / dismissed cases -> new thresholds and/or new training labels.

The backend exports a CSV (one row per analyst decision):

    txn_id,decision[,analyst,decided_at]
    TXN-0027580,confirmed
    TXN-0024183,dismissed

    confirmed = analyst says it IS fraud  (label 1)      dismissed = false alarm  (label 0)

Two ways the feedback is used:
  1. THRESHOLDS (instant, no retraining): re-tune the otp / hold / block cut-offs with the same cost
     model as ml/decision.py, but analyst-reviewed cases count `--weight` times more than the
     historical validation rows. Safety rails: needs >= --min-cases decisions, and each threshold may
     move at most --max-shift away from its current value per update.
  2. LABELS (slower, better model): write data/feedback_labels.csv; the next `python -m ml.train`
     overrides the TRAINING labels of those transactions (test labels are never touched). This is
     also how unreported fraud, which the historical labels missed, gets corrected.

Usage:
    python -m ml.feedback feedback.csv                 # dry run: show suggested thresholds
    python -m ml.feedback feedback.csv --apply         # save thresholds into models/risk_engine.joblib + rebuild cache
    python -m ml.feedback feedback.csv --labels        # also write data/feedback_labels.csv for retraining
    python -m ml.feedback --simulate 300               # demo: fake analysts review random flagged test cases
    python -m ml.feedback --simulate 120 --drift --apply   # demo: analysts dismiss borderline alerts -> thresholds rise
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.decision import tune_thresholds, total_cost  # noqa: E402

LABEL_FILE = ROOT / "data" / "feedback_labels.csv"
CONFIRM = {"confirmed", "confirm", "confirmed_fraud", "fraud", "true_positive", "1"}
DISMISS = {"dismissed", "dismiss", "false_positive", "legit", "not_fraud", "0"}


def load_feedback(path) -> pd.DataFrame:
    fb = pd.read_csv(path)
    d = fb["decision"].astype(str).str.strip().str.lower()
    unknown = ~d.isin(CONFIRM | DISMISS)
    if unknown.any():
        print(f"warning: ignoring {int(unknown.sum())} rows with unrecognised decision values")
    fb = fb[~unknown].copy()
    fb["is_fraud"] = d[~unknown].isin(CONFIRM).astype(int)
    return fb.drop_duplicates("txn_id", keep="last")[["txn_id", "is_fraud"]].reset_index(drop=True)   # latest decision wins


def reference_rows(cache: pd.DataFrame) -> pd.DataFrame:
    """Historical labelled rows the model did NOT train on (same slice train.py uses for tuning)."""
    tr = cache[cache.split == "train"]
    return tr[tr.ts >= tr.ts.quantile(0.8)]


def adapt_thresholds(cache, fb, base_th, weight=5.0, min_cases=30, max_shift=0.10):
    ref = reference_rows(cache)
    ref = ref[~ref.txn_id.isin(fb.txn_id)]                      # analyst label replaces the old one
    rev = cache.merge(fb, on="txn_id", suffixes=("_old", ""))
    info = dict(n_feedback=len(rev), n_confirmed=int(rev.is_fraud.sum()), n_dismissed=int((rev.is_fraud == 0).sum()))
    if len(rev) < min_cases:
        return list(base_th), dict(info, changed=False, reason=f"only {len(rev)} reviewed cases (< {min_cases}); thresholds kept")

    score = np.r_[ref.risk_score.values, rev.risk_score.values]
    amount = np.r_[ref.amount.values, rev.amount.values]
    y = np.r_[ref.is_fraud.values, rev.is_fraud.values]
    w = np.r_[np.ones(len(ref)), np.full(len(rev), weight)]

    new = np.array(tune_thresholds(score, amount, y, w))
    base = np.array(base_th)
    new = np.clip(new, base - max_shift, base + max_shift).clip(0.01, 0.99)
    new = np.maximum.accumulate(new)                            # keep otp <= hold <= block
    cost_before, cost_after = (total_cost(score, amount, y, t, w) for t in (base, new))
    return [round(float(t), 2) for t in new], dict(info, changed=bool((new != base).any()),
                                                  weighted_cost_before=round(float(cost_before)),
                                                  weighted_cost_after=round(float(cost_after)))


def write_training_labels(fb, path=LABEL_FILE):
    old = pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=["txn_id", "is_fraud"])
    out = pd.concat([old, fb]).drop_duplicates("txn_id", keep="last")
    out.to_csv(path, index=False)
    return len(out)


def simulate(cache, n, seed=0, drift=False):
    """Demo only: pretend analysts review n random flagged TEST cases and are always right
    (truth = injected scenario, so this also fixes the ~6% unreported-fraud label noise).
    drift=True: analysts keep dismissing borderline alerts (risk < 0.5), e.g. a legit shopping wave."""
    pool = cache[(cache.split == "test") & (cache.action != "allow")]
    if drift:
        pool = pool[pool.risk_score < 0.5]
    pick = pool.sample(min(n, len(pool)), random_state=seed)
    decision = np.where(pick.scenario.values != "normal", "confirmed", "dismissed")
    if drift:
        decision = np.full(len(pick), "dismissed")
    return pd.DataFrame(dict(txn_id=pick.txn_id.values, decision=decision))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feedback_csv", nargs="?")
    ap.add_argument("--simulate", type=int, metavar="N", help="demo: generate N fake analyst decisions")
    ap.add_argument("--drift", action="store_true", help="with --simulate: analysts dismiss borderline alerts (shows thresholds adapting)")
    ap.add_argument("--apply", action="store_true", help="save new thresholds into the model artifact and rebuild the cache")
    ap.add_argument("--labels", action="store_true", help="also write data/feedback_labels.csv for the next retrain")
    ap.add_argument("--weight", type=float, default=5.0)
    ap.add_argument("--min-cases", type=int, default=30)
    ap.add_argument("--max-shift", type=float, default=0.10)
    a = ap.parse_args()

    from ml.cache import build_cache, load_cache
    cache = load_cache()
    if a.simulate:
        sim = simulate(cache, a.simulate, drift=a.drift)
        path = ROOT / "data" / "cache" / "feedback_simulated.csv"      # git-ignored
        sim.to_csv(path, index=False)
        print(f"simulated {len(sim)} analyst decisions -> {path.relative_to(ROOT)}")
        a.feedback_csv = path
    if not a.feedback_csv:
        ap.error("give a feedback CSV or use --simulate N")

    fb = load_feedback(a.feedback_csv)
    art_path = ROOT / "models" / "risk_engine.joblib"
    art = joblib.load(art_path)
    new_th, info = adapt_thresholds(cache, fb, art["thresholds"], a.weight, a.min_cases, a.max_shift)

    print(f"feedback: {info['n_feedback']} cases ({info['n_confirmed']} confirmed, {info['n_dismissed']} dismissed)")
    print(f"thresholds (otp, hold, block): {art['thresholds']}  ->  {new_th}")
    if "weighted_cost_before" in info:
        print(f"weighted cost: BDT {info['weighted_cost_before']:,} -> {info['weighted_cost_after']:,}")
    else:
        print(info["reason"])

    if a.labels:
        print(f"wrote {write_training_labels(fb)} analyst labels to {LABEL_FILE.relative_to(ROOT)} (used by the next `python -m ml.train`)")
    if a.apply and info["changed"]:
        art["thresholds"] = new_th
        joblib.dump(art, art_path)
        build_cache()
        print("applied: model artifact updated and scored cache rebuilt")
    elif a.apply:
        print("nothing to apply")
    elif info["changed"]:
        print("dry run - add --apply to save these thresholds")


if __name__ == "__main__":
    main()
