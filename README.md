# UpayShield

**Case-centric fraud and risk intelligence for mobile money.** Every alert answers three questions:
**What happened? Why is it risky? What should Upay do next?**

> Hackathon Track 01: Trust & Risk Intelligence. This README currently documents the **AI/ML module (v1)**.
> Backend and frontend sections will be added by the rest of the team.

## ML module: version 1

| Step | What it does |
|---|---|
| Data | Own synthetic sample dataset in `data/` with a time-based train/test split |
| Features | 23 point-in-time features: behaviour deviation, account-takeover signals, mule/network signals, agent signals |
| Models | Logistic Regression, Random Forest, LightGBM (supervised) and Isolation Forest (unsupervised baseline) |
| Selection | Best supervised model chosen on a **validation** slice (last 20% of train), never on test |
| Decision engine | Risk score maps to `allow / otp_step_up / hold / block`, with thresholds tuned on cost (fraud loss vs. customer friction) |
| Output | `RiskEngine.score_frame()` returns what-happened / why-risky / what-next JSON for the API |

### Fraud scenarios covered
Account takeover (ATO), scam victims, mule pass-through rings, structuring (just under 50,000 BDT) and rogue agents.

## Setup and run

```bash
git clone https://github.com/Ridwan-Rythm/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ml.train                 # trains, compares models, writes reports/ and models/
python -m ml.score                 # demo: prints the 3 riskiest test transactions as case JSON
```

The dataset is already in `data/`. To regenerate it (same seed gives the same data):
`python scripts/generate_data.py --seed 42`

## Results (test set, last 30% of the timeline)

| Model | ROC-AUC | PR-AUC | Precision@3% | Recall@3% |
|---|---|---|---|---|
| Logistic Regression | 0.9982 | 0.9119 | 0.8542 | 0.9248 |
| Random Forest | 0.9979 | 0.8975 | 0.8611 | 0.9323 |
| LightGBM | 0.9985 | 0.9314 | 0.8819 | 0.9549 |
| Isolation Forest (unsupervised) | 0.9777 | 0.6655 | 0.6597 | 0.7143 |

Business impact of the selected model with tuned actions: about **94% of fraud value prevented**, friction on
about **1.5% of legitimate transactions**. Full numbers are in `reports/metrics_v1.json`; the chart is
`reports/model_comparison_v1.png`.

## Limitations (honest notes)
- The data is synthetic, so scores are optimistic. Use them to compare models, not as real-world performance.
- Scam-victim payments are the hardest scenario (about 81% recall): the victim's own device and habits look normal.
- Action stop-rates and friction costs in `ml/decision.py` are assumptions to tune with the team.
- Validation picked Random Forest while LightGBM scored higher on test; the gap is small and within noise on this data size.

## Roadmap
- **v2:** SHAP explanations, cross-validation and hyperparameter tuning, probability calibration
- **v3:** graph features (NetworkX) for mule rings, agent peer-comparison model
- **v4:** feedback loop / retraining from analyst decisions, drift checks
- **Final:** integrated with FastAPI `/score`, dashboard and LLM investigation assistant

## Repository layout
```
data/      train.csv, test.csv, README.md
scripts/   generate_data.py
ml/        features.py, decision.py, train.py, score.py
reports/   metrics, model comparison table and chart
```

## Team
- AI/ML: Ridwan Rythm
- Backend / Intelligence: _TBD_
- Frontend / Product: _TBD_
