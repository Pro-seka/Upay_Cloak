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
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Setup & Run](#setup--run)
8. [API Contract](#api-contract)
9. [Risk Scoring & Decision Policy](#risk-scoring--decision-policy)
10. [Demo Walkthrough](#demo-walkthrough)
11. [Data](#data)
12. [Evaluation Results](#evaluation-results)
13. [Project Status](#project-status)
14. [Limitations & Future Work](#limitations--future-work)
15. [Team & Contributions](#team--contributions)

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
Data (PaySim backbone + synthetic enrichment)
  -> Feature pipeline
  -> LightGBM + Isolation Forest + Graph analytics + Agent peer model
  -> Risk engine (combined score, rule trace, reasons, tags)
  -> Decision engine (recommended action)
  -> Evidence builder (structured JSON) -> LLM narrative (template fallback)
  -> FastAPI -> Dashboard (queue, case view, graph, stream, warning demo)
```

The frontend talks to the backend only through `assets/js/api.js`. A single flag (`USE_MOCK`) switches between built-in mock data and the real API, so the UI and the backend can be developed independently.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | Python, LightGBM, scikit-learn (Isolation Forest), SHAP |
| Graph | NetworkX |
| Backend | FastAPI, DuckDB / SQLite |
| AI assistant | LLM API grounded in evidence JSON, with template fallback |
| Frontend | Vanilla HTML, CSS and JavaScript, Chart.js, vis-network |
| Fonts | Plus Jakarta Sans, Noto Sans Bengali |

---

## Project Structure

```
.
├── data/                 # README only, no raw data is committed
├── scripts/              # download_data.py, generate_data.py
├── ml/                   # features, models, evaluation
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

### 1. Clone

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Run the dashboard (works today, with built-in mock data)

The frontend is static, so any simple web server works:

```bash
cd frontend
python -m http.server 8000
```

Open <http://localhost:8000>. No build step and no install needed.

> Fonts load from Google Fonts. Chart.js and vis-network are bundled in `assets/js/vendor/`.

### 3. Run with the real backend

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env               # add your LLM key (optional)

# Generate data and train models
python scripts/generate_data.py
python ml/train.py

# Start the API
uvicorn backend.main:app --reload --port 8001
```

Then in `frontend/assets/js/api.js` set:

```js
const USE_MOCK = false, BASE_URL = '/api';
```

and serve the frontend so `/api` reaches the backend (reverse proxy, or point `BASE_URL` at `http://localhost:8001`).

> The exact backend and ML commands above will be finalized as those modules land. See [Project Status](#project-status).

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
| 0.65 to 0.79 | High | **Step-up**: customer must confirm with a PIN |
| 0.80 and above | Critical | **Block** |

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

We use a **synthetic, Bangladesh-flavored dataset**: BDT amounts, Dhaka/Chattogram-style locations, mobile-money transaction types, and Eid and salary-day spikes.

- **Backbone:** PaySim-style schema and amount/type distributions.
- **Enrichment layer (ours):** device IDs, locations, timestamps with per-user habits, agent IDs with peer groups, and recipient history.
- **Injected, labeled scenarios:** account takeover, mule rings, scam victims, rogue agents, structuring.
- **Honest metrics:** noise and label ambiguity are added so results are not unrealistically perfect.

Raw datasets are **never committed**. See `data/README.md` and `scripts/download_data.py`.

---

## Evaluation Results

Results will be filled in after model training.

| Metric | Value |
|---|---|
| ROC-AUC | TBD |
| PR-AUC | TBD |
| Precision / Recall @ K | TBD |
| False-positive rate | TBD |
| Money saved (BDT) | TBD |
| Analyst workload reduction | TBD |

---

## Project Status

| Area | Status |
|---|---|
| Dashboard, case view, graph explorer, warning demo (EN/BN) | Done, running on mock data |
| Frontend/backend API contract | Defined, switchable via `USE_MOCK` |
| Data generator and features | In progress |
| LightGBM, anomaly model, SHAP | In progress |
| FastAPI, graph analytics, agent risk, decision engine | In progress |
| Evidence builder and LLM assistant | In progress |

Milestone tags: `v0.1-data`, `v0.2-e2e`, `v1.0-final`.

---

## Limitations & Future Work

- Scores on synthetic data are not a guarantee of real-world performance.
- The mock-mode assistant uses templates, not a live LLM.
- Login and logout are not part of this demo.
- **Future:** drift monitoring, a Wallet 360 profile page, mule-ring takedown simulation ("what if we freeze these 5 wallets?"), a feedback loop that adjusts thresholds or triggers retraining, and exportable PDF case reports.

---

## Team & Contributions

| Member | Role | Owns |
|---|---|---|
| **Ridwan Siddque** | ML / Data | Data generator, features, LightGBM, anomaly model, SHAP, evaluation |
| **Sakib Hasan** | Backend / Intelligence | FastAPI, graph analytics, agent risk, decision engine, evidence builder, LLM assistant |
| **Aritro Das** | Frontend / Product | Dashboard, graph view, case page, warning demo, README, demo script |



---

## License

Add a license before publishing (for example MIT).
