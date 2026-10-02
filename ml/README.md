# UpayShield ML module (prototype)

Run from the repo root:

    pip install -r requirements.txt
    python scripts/generate_data.py      # synthetic BD mobile-money data + injected fraud
    python -m ml.train                   # features -> LightGBM + Isolation Forest -> thresholds -> metrics

Outputs: `models/risk_engine.joblib`, `reports/metrics.json`, `reports/shap_importance.json`, `reports/sample_cases.json`.

| File | Role |
|---|---|
| `scripts/generate_data.py` | Synthetic data: ATO, scam victims, mule rings, structuring, rogue agents + label noise |
| `ml/features.py` | Point-in-time features (no leakage): behaviour deviation, ATO, network, agent |
| `ml/anomaly.py` | Isolation Forest behavioural anomaly score |
| `ml/rules.py` | Rule trace + scam/ATO/mule tags |
| `ml/explain.py` | SHAP top reasons in plain language |
| `ml/decision.py` | Cost-tuned allow / OTP / hold / block thresholds |
| `ml/evaluate.py` | AUC, PR-AUC, precision/recall@K, money saved, friction, per-scenario recall |
| `ml/score.py` | `RiskEngine.score_frame()` returns what-happened / why-risky / what-next JSON for the API |

## Honest caveats
- Data is synthetic, so metrics are optimistic (ROC-AUC ~0.999). Do not present them as real-world performance.
- `user_txn_count` ranks high in SHAP because mule accounts are brand new in the generator (a data artifact).
- Isolation Forest does not beat LightGBM alone on this data; it is kept as an unsupervised safety net for unseen fraud types.
- Action effectiveness and friction costs in `decision.py` are assumptions to tune with the team.
