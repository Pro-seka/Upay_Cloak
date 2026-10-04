"""Pre-computed scored cache so the backend boots instantly.

Computing the 23 point-in-time features is a sequential replay of every transaction (seconds to
minutes at scale). Do it ONCE here; the backend then just reads a parquet file.

    python -m ml.cache                       # (re)build  data/cache/scored_cache.parquet
    from ml.cache import load_cache          # backend startup: loads (builds only if missing)

Columns: all raw transaction columns + the 23 model features + split (train/test) + risk_score,
anomaly_score, action, tags (comma-separated rule tags) and reasons (JSON string with the SHAP
top-positive / top-negative factors; filled for flagged rows only, empty for "allow").
Rebuild after every retrain, because the scores depend on the model.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.data import DATA_PATH, load_transactions  # noqa: E402
from ml.features import build_features  # noqa: E402
from ml.score import MODEL_PATH, RiskEngine, rule_trace  # noqa: E402

CACHE_PATH = ROOT / "data" / "cache" / "scored_cache.parquet"


def build_cache(csvs=None, model_path=MODEL_PATH, out=CACHE_PATH):
    t0 = time.time()
    raw = load_transactions(csvs)                    # one dataset, time-based train/test `split` column
    feats = build_features(raw)                      # point-in-time, over the whole timeline in order
    eng = RiskEngine(model_path)
    risk, anomaly, actions = eng.predict(feats)
    feats["risk_score"], feats["anomaly_score"], feats["action"] = risk.round(5), anomaly.round(5), actions
    feats["tags"] = [",".join(sorted({h["tag"] for h in rule_trace(r)})) for _, r in feats.iterrows()]

    flagged = feats.index[feats.action != "allow"]   # SHAP only where an analyst will look
    reasons = pd.Series("", index=feats.index, dtype=object)
    if len(flagged):
        expl = eng.explainer.explain(feats.loc[flagged])
        reasons.loc[flagged] = [json.dumps(e) for e in expl]
    feats["reasons"] = reasons

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"cache built: {len(feats):,} txns ({len(flagged):,} flagged with SHAP reasons) "
          f"-> {out.relative_to(ROOT)} [{out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s]")
    return feats


def load_cache(path=CACHE_PATH, rebuild_if_missing=True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if not rebuild_if_missing:
            raise FileNotFoundError(f"{path} missing - run: python -m ml.train  (or python -m ml.cache)")
        print("scored cache not found, building it once ...")
        build_cache(out=path)
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the scored cache")
    ap.add_argument("--csv", nargs="+", help="CSV file(s) to score (default: data/transactions.csv)")
    build_cache(ap.parse_args().csv)
