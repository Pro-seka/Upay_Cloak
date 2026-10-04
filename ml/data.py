"""Single entry point for loading transactions (one dataset, one time-based split).

Default file: data/transactions.csv. The first TRAIN_FRAC of the timeline is "train", the rest "test".
Works with any CSV that has the core columns: a missing `scenario` becomes "normal",
a missing `is_fraud` becomes 0 and a missing `agent_id` stays empty, so real / Kaggle data loads too.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "transactions.csv"
TRAIN_FRAC = 0.70
REQUIRED = ["txn_id", "ts", "user_id", "type", "amount", "recipient_id", "device_id", "location", "balance_before"]


def load_transactions(paths=None) -> pd.DataFrame:
    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else (paths or [DATA_PATH]))]
    df = pd.concat([pd.read_csv(p, parse_dates=["ts"]) for p in paths], ignore_index=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{', '.join(map(str, paths))}: missing required columns {missing}")
    if "is_fraud" not in df:
        df["is_fraud"] = 0
    if "scenario" not in df:
        df["scenario"] = "normal"
    df["scenario"] = df["scenario"].fillna("normal")
    if "agent_id" not in df:
        df["agent_id"] = None
    df = df.sort_values(["ts", "txn_id"], kind="stable").reset_index(drop=True)
    cut = int(len(df) * TRAIN_FRAC)
    df["split"] = ["train"] * cut + ["test"] * (len(df) - cut)      # time-based, never shuffled
    return df
