# UpayShield

**A case-centric fraud and trust-risk platform for mobile financial services. Every alert tells an analyst what happened, why it is risky, and what to do next.**

Built for the **Upay Ai Dev Fest, Track 01: Trust & Risk Intelligence**

> The dashboard UI is branded **Upay_cloak**

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Our Approach](#our-approach)
3. [Features](#features)
4. [Architecture](#architecture)
5. [ML Module (v1)](#ml-module-v1)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Setup & Run](#setup--run)
9. [API Contract](#api-contract)
10. [Risk Scoring & Decision Policy](#risk-scoring--decision-policy)
11. [Demo Walkthrough](#demo-walkthrough)
12. [Data](#data)
13. [Evaluation Results](#evaluation-results)
14. [Project Status](#project-status)
15. [Limitations](#limitations)
16. [Roadmap](#roadmap)
17. [Team & Contributions](#team--contributions)

---

## The Problem

Mobile money platforms face account takeover, money-mule rings, scam victims, and rogue agents, all at the same time. Most fraud tools produce a bare score and leave analysts to piece together the story. That is slow, hard to audit, and creates friction for honest customers.

## Our Approach

UpayShield is built around the **case**, not the model. Every alert answers three questions:

| Question | Answered by |
|---|---|
| **1. What happened?** | Transaction timeline with evidence IDs |
| **2. Why is it risky?** | Model score, SHAP-style reason contributions, behavioral deviation vs. the user's baseline, network evidence, rule trace |
| **3. What should Upay do next?** | Recommended action: allow, warn, step-up verification, hold, block, KYC re-verify, or escalate |

---

## Features

### Core modules

| # | Module | Approach |
|---|---|---|
| 1 | Real-time transaction risk scoring | LightGBM behind a FastAPI `/score` endpoint |
| 2 | Behavioral anomaly detection | Per-user baseline (amount z-score, usual hours, usual recipients) plus Isolation Forest |
| 3 | Account takeover (ATO) signals | New device, new location, odd timing, new recipient, velocity bursts |
| 4 | Mule / network discovery | NetworkX graph: fan-in/out, rapid pass-through, shared devices, community detection |
| 5 | Agent risk | Each agent compared to peers (cash-in/out ratio, volume spikes, structuring) |
| 6 | Scam intelligence | First-time recipient with large amount, rapid repeats, victim-to-mule flows |
| 7 | AI investigation assistant | LLM grounded only in structured evidence JSON, with template fallback |
| 8 | Explainability | Top contributing factors plus rule trace for every score |

### Differentiators

- **Action recommendation engine.** Risk tiers map to concrete actions.
- **Bangla + English customer scam warning.** A pre-send nudge in the customer's own language.
- **Analyst case queue.** Prioritized by risk, filterable by alert type, with a full audit trail of actions.
- **Live transaction stream.** Transactions are scored and appear in real time, with pause and reset controls.
- **Evidence-linked narratives.** Every assistant answer cites evidence IDs (e.g. `E1`, `E2`), so it is auditable and grounded.
- **Network graph explorer.** Search a wallet, choose 1 to 3 hops, filter by risk or entity type, and highlight suspicious rings.
- **Business-impact metrics.** Money protected, false-positive rate, and scoring latency on the dashboard.

### Dashboard pages

| Page | What it does |
|---|---|
| **Dashboard** (`index.html`) | KPIs, alerts over time, top risk reasons, live transactions, risk mix, alert queue, agents vs. peers |
| **Case** (`case.html`) | Three-part case view (what happened / why risky / what next), timeline, reason bars, baseline comparison, related graph, assistant chat, action buttons, audit trail, copy summary |
| **Graph** (`graph.html`) | Interactive network of wallets, agents, merchants and devices to find mule rings |
| **Warning demo** (`demo.html`) | Customer phone view (English and Bangla) next to the scoring "behind the scenes" |
| **Settings / Help** | Analyst profile, language, stream speed, and a plain-language guide to how the system works |

---

## Architecture

```
Data (own synthetic dataset, PaySim-style schema + enrichment)
  -> Feature pipeline
  -> LightGBM + Isolation Forest + Graph analytics + Agent peer model
  -> Risk engine (combined score, rule trace, reasons, tags)
  -> Decision engine (recommended action)
  -> Evidence builder (structured JSON) -> LLM narrative (template fallback)
  -> FastAPI -> Dashboard (queue, case view, graph, stream, warning demo)
```

The frontend talks to the backend only through `assets/js/api.js`. A single flag (`USE_MOCK`) switches between built-in mock data and the real API, so the UI and the backend can be developed independently.

---

## ML Module (v1)

The AI/ML module is implemented and runnable today.

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

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | Python, LightGBM, scikit-learn (Logistic Regression, Random Forest, Isolation Forest), SHAP (planned for v2) |
| Graph | NetworkX |
| Backend | FastAPI, DuckDB / SQLite |
| AI assistant | LLM API grounded in evidence JSON, with template fallback |
| Frontend | Vanilla HTML, CSS and JavaScript, Chart.js, vis-network |
| Fonts | Plus Jakarta Sans, Noto Sans Bengali |

---

## Project Structure

```
.
├── data/                 # train.csv, test.csv, README.md (small synthetic sample)
├── scripts/              # generate_data.py (and download_data.py)
├── ml/                   # features.py, decision.py, train.py, score.py
├── models/               # trained model artifacts (written by ml.train)
├── reports/              # metrics_v1.json, model comparison table and chart
├── backend/              # FastAPI, graph analytics, decision engine, evidence builder
├── frontend/             # dashboard (the trust-radar app)
│   ├── index.html        # dashboard
│   ├── case.html         # case investigation
│   ├── graph.html        # network graph
│   ├── demo.html         # customer warning demo
│   ├── settings.html
│   ├── help.html
│   └── assets/
│       ├── css/style.css
│       └── js/           # api.js, data.js (mock), dashboard.js, case.js,
│                         # graph.js, demo.js, i18n.js, layout.js, utils.js, vendor/
├── docs/                 # architecture notes, screenshots
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup & Run

### Prerequisites

- Python 3.10+ (for the backend and ML pipeline)
- A modern browser
- Optional: an LLM API key (the assistant falls back to templates without one)

### 1. Clone and install

```bash
git clone https://github.com/Ridwan-Rythm/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the ML pipeline

The dataset is already in `data/`.

```bash
python -m ml.train                 # trains, compares models, writes reports/ and models/
python -m ml.score                 # demo: prints the 3 riskiest test transactions as case JSON
```

To regenerate the dataset (the same seed gives the same data):

```bash
python scripts/generate_data.py --seed 42
```

### 3. Run the dashboard (works today, with built-in mock data)

The frontend is static, so any simple web server works:

```bash
cd frontend
python -m http.server 8000
```

Open <http://localhost:8000>. No build step and no install needed.

> Fonts load from Google Fonts. Chart.js and vis-network are bundled in `assets/js/vendor/`.

### 4. Run with the real backend

```bash
cp .env.example .env               # add your LLM key (optional)
uvicorn backend.main:app --reload --port 8001
```

Then in `frontend/assets/js/api.js` set:

```js
const USE_MOCK = false, BASE_URL = '/api';
```

and serve the frontend so `/api` reaches the backend (reverse proxy, or point `BASE_URL` at `http://localhost:8001`).

> The backend commands will be finalized as those modules land. See [Project Status](#project-status).

---

## API Contract

The frontend and backend agree on these endpoints (defined in `frontend/assets/js/api.js`):

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/overview` | KPIs, risk distribution, alert trend, top reasons, agent scores |
| `GET` | `/transactions` | Recent scored transactions (live feed) |
| `POST` | `/score` | Score one transaction, returns risk score, level, decision and reasons |
| `GET` | `/cases` | Alert queue, optionally filtered by `alert_type` |
| `GET` | `/cases/{id}` | Full case: timeline, reasons, baseline, actions, subgraph, narrative |
| `POST` | `/cases/{id}/action` | Apply an analyst action (hold, block, kyc, escalate, false_positive) |
| `POST` | `/cases/{id}/ask` | Ask the investigation assistant, returns text plus evidence IDs |
| `GET` | `/graph/{entity_id}?depth=N` | Network neighborhood around a wallet, agent or device |

**Transaction**

```json
{
  "id": "TXN-1001", "timestamp": "...", "type": "send_money", "amount_bdt": 48000,
  "sender": "017XX-XXX482", "receiver": "018XX-XXX771", "channel": "app",
  "device_id": "DEV-NEW-93", "location": "Chattogram",
  "risk_score": 0.86, "risk_level": "critical", "decision": "block",
  "reasons": [{ "feature": "dev", "label": "New device", "contribution": 0.28, "source": "rule" }]
}
```

**Case**

```json
{
  "case_id": "CASE-1001", "status": "open", "severity": "critical",
  "alert_type": "account_takeover", "entity": { "type": "wallet", "id": "017XX-XXX482" },
  "what_happened": "...", "why_risky": [], "baseline": {},
  "recommended_actions": [], "timeline": [], "related_transactions": [],
  "subgraph": { "nodes": [], "edges": [] }, "narrative": "...", "confidence": 0.9
}
```

Reason `source` is one of `model`, `rule`, `anomaly`, or `graph`.

---

## Risk Scoring & Decision Policy

Each transaction gets a risk score from **0 to 1**. Every signal shows how much it raised or lowered the score.

| Risk score | Level | Decision |
|---|---|---|
| below 0.30 | Low | **Allow** |
| 0.30 to 0.64 | Medium | **Warn** the customer (a risky cash-out is **held**) |
| 0.65 to 0.79 | High | **Step-up**: customer must confirm with a PIN or OTP |
| 0.80 and above | Critical | **Block** |

The ML decision engine (`ml/decision.py`) outputs `allow / otp_step_up / hold / block`, with thresholds tuned on cost (fraud loss vs. customer friction). The platform layer adds customer-facing **warn** and the analyst actions below.

Analysts can then **Hold**, **Block**, **Request KYC**, **Escalate**, or **Mark as false positive**. Every action is saved to an audit trail.

### Alert types

- **Account takeover:** someone else is using a customer's account, often from a new device.
- **Mule network:** many accounts pass stolen money to one wallet, which cashes out fast.
- **Scam:** a customer is tricked into sending money.
- **Agent anomaly:** an agent's activity is far from what similar agents do.

---

## Demo Walkthrough

1. **Live stream.** Open the Dashboard. Transactions are scored in real time and a suspicious one raises an alert.
2. **Open the case.** See what happened, the risk score, the reason bars, and how the behavior deviates from the customer's baseline.
3. **Mule ring.** Open the Graph and click *Highlight suspicious ring* to see the fan-in and cash-out pattern.
4. **AI assistant.** Ask the assistant to summarize the case. The answer cites evidence IDs.
5. **Analyst action.** Click *Hold*, *Block* or *Mark as false positive*. The audit trail updates.
6. **Customer warning.** Open *Warning demo*, pick a scenario, and switch between English and Bangla to see what the customer sees before sending money.
7. **Impact.** Close on the dashboard KPIs: money protected versus false-positive friction.

### Built-in demo scenarios

| Case | Scenario |
|---|---|
| `CASE-1001` | Account takeover: new phone in Chattogram, PIN reset, balance drained in 2 minutes |
| `CASE-1002` | Money-mule ring: 14 wallets fan in, then fast cash-out through one agent |
| `CASE-1003` | Scam victim: first-time, round-amount transfer after a long call from an unknown number |
| `CASE-1004` | Agent anomaly: agent processing 9x the peer median volume |
| `CASE-1005` | Normal behavior: routine transfer, allowed with no friction |

---

## Data

We use a **synthetic, Bangladesh-flavored dataset**: BDT amounts, Dhaka/Chattogram-style locations, mobile-money transaction types, and Eid and salary-day spikes. A small generated sample ships in `data/` (`train.csv`, `test.csv`) with a time-based split.

- **Backbone:** PaySim-style schema and amount/type distributions.
- **Enrichment layer (ours):** device IDs, locations, timestamps with per-user habits, agent IDs with peer groups, and recipient history.
- **Injected, labeled scenarios:** account takeover, mule rings, scam victims, rogue agents, structuring.
- **Honest metrics:** noise and label ambiguity are added so results are not unrealistically perfect.

Third-party raw datasets are **never committed**. See `data/README.md` and `scripts/generate_data.py`.

---

## Evaluation Results

Test set = last 30% of the timeline.

| Model | ROC-AUC | PR-AUC | Precision@3% | Recall@3% |
|---|---|---|---|---|
| Logistic Regression | 0.9982 | 0.9119 | 0.8542 | 0.9248 |
| Random Forest | 0.9979 | 0.8975 | 0.8611 | 0.9323 |
| **LightGBM** | **0.9985** | **0.9314** | **0.8819** | **0.9549** |
| Isolation Forest (unsupervised) | 0.9777 | 0.6655 | 0.6597 | 0.7143 |

**Business impact** of the selected model with tuned actions: about **94% of fraud value prevented**, with friction on about **1.5% of legitimate transactions**. Full numbers are in `reports/metrics_v1.json`; the chart is `reports/model_comparison_v1.png`.

| Metric | Value |
|---|---|
| ROC-AUC (LightGBM) | 0.9985 |
| PR-AUC (LightGBM) | 0.9314 |
| Precision / Recall @ 3% | 0.8819 / 0.9549 |
| Fraud value prevented | about 94% |
| Friction on legitimate transactions | about 1.5% |
| Money saved (BDT) | TBD |
| Analyst workload reduction | TBD |

---

## Project Status

| Area | Status |
|---|---|
| Dashboard, case view, graph explorer, warning demo (EN/BN) | Done, running on mock data |
| Frontend/backend API contract | Defined, switchable via `USE_MOCK` |
| Data generator and features (23 features) | Done (v1) |
| Supervised models, Isolation Forest, decision engine | Done (v1) |
| SHAP explanations, calibration, tuning | Planned (v2) |
| FastAPI, graph analytics, agent risk | In progress |
| Evidence builder and LLM assistant | In progress |

Milestone tags: `v0.1-data`, `v0.2-e2e`, `v1.0-final`.

---

## Limitations

- The data is synthetic, so scores are optimistic. Use them to compare models, not as real-world performance.
- Scam-victim payments are the hardest scenario (about 81% recall): the victim's own device and habits look normal.
- Action stop-rates and friction costs in `ml/decision.py` are assumptions to tune with the team.
- Validation picked Random Forest while LightGBM scored higher on test; the gap is small and within noise at this data size.
- The mock-mode assistant uses templates, not a live LLM.
- Login and logout are not part of this demo.

## Roadmap

- **v2:** SHAP explanations, cross-validation and hyperparameter tuning, probability calibration
- **v3:** graph features (NetworkX) for mule rings, agent peer-comparison model
- **v4:** feedback loop / retraining from analyst decisions, drift checks
- **Final:** integration with FastAPI `/score`, the dashboard, and the LLM investigation assistant
- **Future ideas:** Wallet 360 profile page, mule-ring takedown simulation ("what if we freeze these 5 wallets?"), threshold adjustment from analyst feedback, exportable PDF case reports

---

## Team & Contributions

| Member | Role | Owns |
|---|---|---|
| **Ridwan Siddque** (GitHub: Ridwan-Rythm) | ML / Data | Data generator, features, LightGBM, anomaly model, SHAP, evaluation |
| **Sakib Hasan** | Backend / Intelligence | FastAPI, graph analytics, agent risk, decision engine, evidence builder, LLM assistant |
| **Aritro Das** | Frontend / Product | Dashboard, graph view, case page, warning demo, README, demo script |

---
