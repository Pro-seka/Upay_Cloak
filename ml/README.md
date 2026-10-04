# UpayShield ML module (v1.1)

Everything the backend needs from the ML side: a trained risk engine, per-transaction explanations,
a pre-scored cache for instant startup, and an analyst-feedback loop.

## Quick start (from the repo root)

```bash
pip install -r requirements.txt          # needs: pandas numpy scikit-learn lightgbm shap pyarrow joblib matplotlib
python -m ml.train                       # reads data/transactions.csv, trains, writes models/ + reports/ AND builds data/cache/scored_cache.parquet
python -m ml.score                       # demo: the 3 riskiest test transactions as full case JSON
```

## Files

| File | Role |
|---|---|
| `data.py` | Loads `data/transactions.csv` and makes the time-based 70/30 `split` column (fills in missing `scenario` / `is_fraud` so other datasets load too) |
| `features.py` | 23 point-in-time features (no future leakage) |
| `rules.py` | The single `rule_trace` (rule hits + tags); `score.py` and `cache.py` import it |
| `anomaly.py` | Isolation Forest behavioural anomaly score (0..1 percentile) |
| `train.py` | Compares Logistic Regression / Random Forest / LightGBM / Isolation Forest, tunes action thresholds, saves the artifact, builds the cache |
| `decision.py` | Risk score to `allow / otp_step_up / hold / block`, thresholds tuned on cost (grid 0.01 to 0.95) |
| `evaluate.py` | Standalone metric helpers (`train.py` has its own copies) |
| `explain.py` | Local factor breakdown: top factors that raised and lowered the risk of any transaction |
| `score.py` | `RiskEngine` for the backend: what-happened / why-risky / what-next JSON |
| `cache.py` | Builds and loads the pre-scored dataset |
| `feedback.py` | Turns analyst Confirm / Dismiss decisions into new thresholds and training labels |

## Model artifact: `models/risk_engine.joblib`

`ml/score.py` loads this by default (`ml.score.MODEL_PATH`). It is a dict with `model`, `anomaly`,
`features`, `thresholds` (`[otp, hold, block]`), plus `name` and `version`. It is git-ignored; every
teammate rebuilds it with `python -m ml.train`.

## Explanations (`ml/explain.py`)

```python
from ml.score import RiskEngine
engine = RiskEngine()
cases = engine.score_frame(features_df)            # SHAP only for flagged rows by default
cases[0]["why_risky"]["feature_contributions"]
# {"base_value": 0.5, "risk": 0.996, "unit": "probability",
#  "top_positive": [{"feature": "amount_z", "value": 7.12, "contribution": 0.1044,
#                    "direction": "increases_risk", "text": "Amount is 7.1 std devs above ..."}, ...],
#  "top_negative": [{"feature": "is_new_recipient", "value": 0.0, "contribution": -0.08,
#                    "direction": "reduces_risk", "text": "Known recipient (paid before)"}, ...]}
```

- `base_value + sum(all contributions) == risk` exactly (verified), so the breakdown is auditable.
- `explain="all"` explains every row, `"none"` skips it (fastest).
- Duplicate-sounding factors (for example `amount` and `log_amount`) are merged into one line.
- **Caveat:** the Random Forest is trained with class weights, so its scores are *ranking scores*, not
  calibrated fraud probabilities, and `base_value` is about 0.5 rather than the real fraud rate. Probability
  calibration is planned for v2.

## Scored cache (`ml/cache.py`)

```python
from ml.cache import load_cache
df = load_cache()      # ~1.7 s; builds once (about 18 s) if the file is missing
```

`data/cache/scored_cache.parquet` holds every transaction with its raw columns, the 23 features, `split`,
`risk_score`, `anomaly_score`, `action`, `tags`, and `reasons` (JSON string with the top positive and negative
factors, filled for flagged rows only). Rebuild it after any retrain or threshold change
(`python -m ml.cache`). For a different dataset: `python -m ml.cache --csv other.csv` (the same 70/30 time split is applied; a missing `scenario` column becomes `normal`).
It is git-ignored.

## Analyst feedback loop (`ml/feedback.py`)

The backend exports one row per analyst decision:

```csv
txn_id,decision
TXN-0027580,confirmed
TXN-0024183,dismissed
```

```bash
python -m ml.feedback feedback.csv                 # dry run: shows the suggested thresholds
python -m ml.feedback feedback.csv --apply         # saves thresholds into the artifact + rebuilds the cache
python -m ml.feedback feedback.csv --labels        # also writes data/feedback_labels.csv for the next retrain
python -m ml.feedback --simulate 120 --drift       # demo without a backend (analysts dismiss borderline alerts)
```

1. **Thresholds (instant):** re-tunes the cut-offs with the same cost model as `decision.py`, with reviewed
   cases counting 5x more than historical rows. Safety rails: at least 30 decisions are required, and each
   threshold moves at most 0.10 per update.
2. **Labels (better model):** `data/feedback_labels.csv` overrides *training* labels on the next
   `python -m ml.train` (test labels are never changed). This also fixes unreported fraud the original labels missed.

## Known limitations
- Synthetic data: use the numbers to compare models, not as real-world performance. The same mule rings appear in train and test, so metrics are optimistic.
- Thresholds are low (about 0.01 to 0.03) because the model is class-weighted. The backend must read them from `models/risk_engine.joblib`, not hard-code them.
- Unknown transaction types (for example PaySim `DEBIT`) get `type_code = 4` instead of crashing.
- Decision-engine stop-rates and friction costs are assumptions.
- The simulated analyst in `--simulate` is always right; real analysts are not, which is why the safety rails exist.
