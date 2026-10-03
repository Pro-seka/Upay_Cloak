# UpayShield (Upay_cloak
**A Case-Centric Trust & Risk Intelligence Platform for Mobile Financial Services**

Built for the **Upay AI Dev Fest — Track 01: Trust & Risk Intelligence**  
*The analyst dashboard and customer-facing UI are branded **Upay_cloak**.*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Features & AI Usage](#2-features--ai-usage)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Requirements & Prerequisites](#5-requirements--prerequisites)
6. [Installation & Setup](#6-installation--setup)
7. [Environment Variables](#7-environment-variables)
8. [Run & Build Commands](#8-run--build-commands)
9. [Live Deployment URL](#9-live-deployment-url)
10. [Testing & Verification Instructions](#10-testing--verification-instructions)
11. [Other Configuration](#11-other-configuration)
12. [Evaluation & Business Impact](#12-evaluation--business-impact)
13. [Conclusion & Roadmap](#13-conclusion--roadmap)
14. [Team & Contributions](#14-team--contributions)

---

## 1. Project Overview

### The Problem
Mobile Financial Services (MFS) platforms like Upay face diverse and sophisticated fraud patterns simultaneously:
- **Account Takeover (ATO):** Fraudsters gain access to legitimate accounts via SIM swaps or social engineering, change devices, and drain balances in seconds.
- **Money-Mule Networks:** Dispersed rings of accounts funnel illicit funds through layered pass-through transfers to central cash-out agents.
- **Scam Victims:** Customers are manipulated into willingly authorizing transfers to fraudsters (lottery scams, fake customer-care calls), where typical authentication fails to protect them.
- **Rogue Agents:** Malicious or compromised cash-in/cash-out agents engage in structuring (smurfing under the 50,000 BDT regulatory threshold) and unusual off-hour transaction volume.

Most conventional fraud tools generate isolated probability scores without context. Fraud analysts are forced to manually piece together disparate logs, leading to slow response times, high operational overhead, inconsistent decisions, and unnecessary friction for honest users.

### The Proposed Solution
**UpayShield** transforms fraud operations from reactive score-checking into a **case-centric investigation and automated decisioning ecosystem**. For every single transaction and alert, UpayShield answers three fundamental questions:

| Question | How UpayShield Answers It |
|---|---|
| **1. What happened?** | A chronological transaction timeline with immutable, auditable evidence IDs (`E1`, `E2`, `E3`). |
| **2. Why is it risky?** | Transparent risk decomposition: LightGBM score, local SHAP-style reason contributions, behavioral deviations vs. the user's historical baseline, graph topology flags, and explicit rule hits. |
| **3. What should Upay do next?** | Recommended policy actions (**Allow**, **Customer Warning**, **Step-up OTP/PIN**, **Hold for Review**, **Block**, **Freeze Wallet**) with full audit trail logging and analyst feedback integration. |

### Purpose of the Project
To provide Upay with a production-grade, end-to-end trust and risk engine that detects emerging threats in real time, slashes fraud loss, prevents customer harm via pre-transaction nudges in **English and Bangla**, and empowers fraud analysts with grounded, explainable AI.

---

## 2. Features & AI Usage

### Core Modules & AI Implementation

| # | Feature / Module | AI / Algorithmic Implementation |
|---|---|---|
| **1** | **Real-Time Transaction Risk Scoring** | High-throughput **LightGBM Classifier** combined with a cost-sensitive decision engine returning risk scores (0.0 to 1.0) and recommended actions within **<15 ms**. |
| **2** | **Behavioral Anomaly Detection** | **Isolation Forest** coupled with rolling per-user statistical baselines (amount z-score, transacting hours, recipient novelty, and device velocity). |
| **3** | **Account Takeover (ATO) Detection** | Multi-signal heuristic and classifier rules detecting abrupt changes in device fingerprint, geo-location hops, late-night activity, and rapid balance drain. |
| **4** | **Mule & Network Ring Discovery** | In-memory **NetworkX Graph Engine** analyzing directed transaction flows, identifying high fan-in/fan-out ratios, rapid pass-through velocity, shared device clusters, and community subgraphs. |
| **5** | **Agent Risk & Structuring Intelligence** | Peer-group anomaly profiling comparing each cash-in/out agent against regional peer distributions, flagging structuring just below 50,000 BDT and off-hour bursts. |
| **6** | **Explainability & Factor Attribution** | Global and local **SHAP-style explainability**, breaking down every flagged score into positive (risk-increasing) and negative (risk-reducing) contributions with human-readable rationale. |
| **7** | **Grounded AI Investigation Assistant** | Dual-mode assistant (Google Gemini / OpenAI / Anthropic or deterministic template engine) strictly grounded in structured `EvidenceBundle` JSON. Every narrative or answer cites immutable evidence IDs (`E1`, `RULE-ATO`, `METRIC-risk_score`) to eliminate hallucination. |
| **8** | **Dual-Language Customer Scam Warning** | Pre-transaction nudge modal supporting both **English and Bangla (বাংলা)**, intercepting scam payments before money leaves the sender's wallet. |
| **9** | **Analyst Case Queue & Audit Trail** | Prioritized queue by risk tier and alert type (`account_takeover`, `mule_network`, `scam`, `agent_anomaly`) with one-click actions (`Hold`, `Block`, `Request KYC`, `Escalate`, `False Positive`) recorded to an immutable audit trail. |
| **10**| **Live Screening Simulation Feed** | Real-time transaction ingestion stream with pause, resume, reset controls, and real-time risk classification gauges. |

---

## 3. Technology Stack

### Backend & Machine Learning
- **Language:** Python 3.10+
- **API Framework:** FastAPI, Starlette, Uvicorn (ASGI)
- **Machine Learning:** LightGBM, scikit-learn (Random Forest, Logistic Regression, Isolation Forest), NumPy, Pandas
- **Explainability:** SHAP (TreeExplainer / LinearExplainer local attribution)
- **Graph Analytics:** NetworkX (Directed Graph, MultiDiGraph, Cycle and Ring Detection)
- **Data & Storage:** PyArrow / Parquet (pre-computed fast-scoring cache), SQLite, Joblib
- **Validation & Settings:** Pydantic v2, Pydantic-Settings
- **LLM Integrations (Optional):** Google Gemini (`gemini-2.0-flash`), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-haiku`) with deterministic template fallback

### Frontend (Dashboard & Mobile Warning Demo)
- **Core:** Vanilla JavaScript (ES6+), HTML5, CSS3 (zero build steps, zero node dependencies)
- **Data Visualizations:** Chart.js (risk distributions, alert trends, feature contributions)
- **Network Graph Rendering:** Vis-network (interactive multi-hop graph explorer)
- **Typography:** Plus Jakarta Sans, Noto Sans Bengali (বাংলা)

### Testing & Code Quality
- **Testing:** PyTest, PyTest-Asyncio, HTTPX / TestClient
- **Linting & Formatting:** Ruff

---

## 4. System Architecture

```
                                 ┌───────────────────────────────┐
                                 │     Transactions Dataset      │
                                 │    (PaySim-style + BDT MFS)   │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │  Point-in-Time Feature Engine │
                                 │   (23 behavioral features)    │
                                 └───────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┼────────────────────────┐
                        ▼                        ▼                        ▼
             ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
             │ LightGBM Classifier │  │  Isolation Forest   │  │   NetworkX Graph    │
             │ (Supervised Risk)   │  │ (Behavior Anomaly)  │  │(Mule Ring Analytics)│
             └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
                        │                        │                        │
                        └────────────────────────┼────────────────────────┘
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │  Risk Engine & Decision Policy│
                                 │  - Tuned Action Thresholds    │
                                 │  - SHAP Factor Contributions  │
                                 │  - Rule & Policy Trace        │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │    FastAPI Unified Backend    │
                                 │   - /api/v1/score, /api/cases │
                                 │   - Grounded AI Assistant     │
                                 │   - Static Frontend Hosting   │
                                 └───────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             ┌─────────────────────┐                           ┌─────────────────────┐
             │  Analyst Dashboard  │                           │ Customer Phone Demo │
             │ (Queue, Cases, Graph│                           │(English & Bangla UI)│
             └─────────────────────┘                           └─────────────────────┘
```

---

## 5. Requirements & Prerequisites

### System Requirements
- **Operating System:** Windows 10/11, macOS (Intel/Apple Silicon), or Linux (Ubuntu 20.04+)
- **Python:** Python 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware:** Standard CPU (minimum 4 GB RAM; no dedicated GPU required)
- **Web Browser:** Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Safari)

---

## 6. Installation & Setup

Follow these exact steps to set up the project on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Ridwan-Rythm/Upay_Cloak.git
cd Upay_Cloak
```

### 2. Create and Activate a Python Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create the Configuration File
Copy the example environment configuration:
```bash
# On Windows PowerShell:
Copy-Item .env.example .env

# On macOS / Linux:
cp .env.example .env
```

---

## 7. Environment Variables

All settings are managed via [`backend/app/config.py`](file:///d:/Upay_Cloak/backend/app/config.py) and read from `.env` or system environment variables. **Every variable includes a safe default, so the application boots and functions with zero configuration.**

| Variable | Type | Default | Description |
|---|---|---|---|
| `UPAY_DATA_PATH` | Path | `data/transactions.csv` | Path to the raw/enriched transactions CSV file. |
| `UPAY_MODEL_PATH` | Path | `models/risk_engine.joblib` | Path to the trained ML model artifact bundle. |
| `UPAY_REPORTS_DIR` | Path | `reports` | Directory where evaluation metrics and comparison charts are saved. |
| `UPAY_CACHE_DIR` | Path | `data/cache` | Directory storing the fast Parquet scoring cache. |
| `UPAY_DB_PATH` | Path | `data/upayshield.db` | SQLite database path for persistent case state and audit logs. |
| `UPAY_AUTO_TRAIN` | Boolean | `false` | If `true`, automatically executes `ml.train` on boot if model artifacts are missing. |
| `UPAY_CORS_ORIGINS` | JSON list | `["*"]` | Allowed HTTP CORS origins for external API access. |
| `UPAY_STREAM_DEFAULT_INTERVAL_MS`| Integer | `3500` | Ingestion speed (ms) for the dashboard live transaction feed. |
| `UPAY_STREAM_DEMO_ALERT_EVERY` | Integer | `6` | Generates a high-priority flagged alert every N simulated stream events. |
| `UPAY_LLM_PROVIDER` | String | `none` | AI Assistant provider: `none` (grounded template fallback), `gemini`, `openai`, or `anthropic`. |
| `UPAY_LLM_API_KEY` | String | *None* | API key for the chosen LLM provider (e.g., Google AI Studio, OpenAI, or Anthropic). |
| `UPAY_LLM_MODEL` | String | *Provider default* | Model name override (e.g. `gemini-2.0-flash` or `gpt-4o-mini`). |
| `UPAY_LLM_TIMEOUT_S` | Float | `8.0` | Maximum network timeout in seconds for LLM generation. |

> **Note on LLM API Keys:** If `UPAY_LLM_PROVIDER=none` or no API key is provided, the assistant automatically uses the built-in deterministic, evidence-grounded template engine. It operates with **zero latency** and **100% reliability**, citing valid evidence IDs.

---

## 8. Run & Build Commands

### Step 1: Train & Build ML Artifacts
The dataset is pre-generated in `data/`. Run the ML pipeline to train models, compute decision thresholds, generate SHAP importances, and compile the instant-boot scoring cache:
```bash
python -m ml.train
```

To quickly verify scoring output on test cases:
```bash
python -m ml.score
```

*(Optional) To re-synthesize the dataset from scratch with seed 42:*
```bash
python scripts/generate_data.py --seed 42
```

### Step 2: Start the Unified Backend & Frontend Server
Launch the unified FastAPI application on port 8000:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### Step 3: Open the Platform
Open your browser and visit:
- **Dashboard & Alert Queue:** [http://localhost:8000](http://localhost:8000)
- **Interactive Network Graph Explorer:** [http://localhost:8000/graph.html](http://localhost:8000/graph.html)
- **Customer Scam Warning Demo (EN/BN):** [http://localhost:8000/demo.html](http://localhost:8000/demo.html)
- **Case Investigation View:** [http://localhost:8000/case.html?id=CASE-1001](http://localhost:8000/case.html?id=CASE-1001)
- **Interactive OpenAPI / Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 9. Live Deployment URL

- **Primary Live Deployment:** [https://upay-cloak.onrender.com](https://upay-cloak.onrender.com) *(or your deployed production URL)*
- **Interactive API Documentation:** [https://upay-cloak.onrender.com/docs](https://upay-cloak.onrender.com/docs)
- **Local Host URL (Judge Testing):** [http://localhost:8000](http://localhost:8000)

> **Deployment Note for Judges:** The backend is fully containerized and cloud-ready. To deploy on Docker, Render, Railway, or Google Cloud Run, execute:
> ```bash
> uvicorn backend.main:app --host 0.0.0.0 --port $PORT
> ```

---

## 10. Testing & Verification Instructions

### 1. Automated Integration Test Suite
The repository includes an automated test suite verifying all core API contracts, frontend routes, graph analytics, case investigations, and actions:
```bash
python -m pytest tests/test_api.py -v
```

**Expected Output:**
```text
tests/test_api.py::test_health PASSED
tests/test_api.py::test_config PASSED
tests/test_api.py::test_frontend_overview PASSED
tests/test_api.py::test_frontend_cases PASSED
tests/test_api.py::test_frontend_case_detail_and_ask PASSED
tests/test_api.py::test_frontend_action_application PASSED
tests/test_api.py::test_frontend_graph PASSED
tests/test_api.py::test_v1_score_endpoint PASSED
tests/test_api.py::test_v1_cases_endpoint PASSED
========================= 9 passed in 3.14s =========================
```

### 2. Manual Verification Workflow (Judge Walkthrough)
1. **Health Verification:** Access `http://localhost:8000/health`. Confirm `{"status": "ok", "model_loaded": true, "intelligence": "live"}`.
2. **Dashboard KPIs:** Navigate to `http://localhost:8000`. Observe live transaction counters, risk mix donut chart, alerts over time, and the top risk reasons bar chart.
3. **Investigate a Case:** In the Alert Queue, click **Investigate** on `CASE-1001` (Account Takeover). Verify:
   - Part 1: Timeline with evidence badges (`E1`, `E2`, `E3`).
   - Part 2: Red/Green SHAP factor contribution bars and baseline comparison table.
   - Part 3: Recommended action, interactive audit trail, and clickable AI Assistant chips.
4. **Interact with AI Assistant:** In the case view sidebar, click *"Why was this flagged?"* or type a question. The response will cite structured evidence IDs.
5. **Explore the Network Graph:** Navigate to `http://localhost:8000/graph.html`. Click **Highlight suspicious ring** to illuminate the multi-wallet mule pass-through cluster.
6. **Customer Warning Demo:** Navigate to `http://localhost:8000/demo.html`. Select the *Scam victim* scenario, click **Send**, and toggle between **English** and **বাংলা** to see the pre-transaction customer intervention.

---

## 11. Other Configuration

### Frontend Mock vs. Live Backend Mode
In [`Frontend/assets/js/api.js`](file:///d:/Upay_Cloak/Frontend/assets/js/api.js), the communication mode is governed by:
```javascript
const USE_MOCK = false, BASE_URL = '/api';
```
- `USE_MOCK = false`: The frontend queries the live Python FastAPI backend and ML models.
- `USE_MOCK = true`: Standalone offline mode using client-side mock datasets.

### Continuous Retraining & Analyst Feedback Loop
UpayShield supports analyst feedback ingestion. When an analyst marks alerts as false positives or confirms fraud, thresholds can be recalibrated using:
```bash
# Simulate 120 analyst decisions with threshold drift and apply to model:
python -m ml.feedback --simulate 120 --drift --apply
```

---

## 12. Evaluation & Business Impact

The models were evaluated using a strict **time-based train/test split** (training on the first 70% of chronological transactions, testing on the unseen final 30%).

### Model Benchmark Comparison

| Model | ROC-AUC | PR-AUC | Precision @ 3% | Recall @ 3% | Inference Latency |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.9982 | 0.9119 | 85.42% | 92.48% | <1 ms |
| Random Forest Classifier | 0.9979 | 0.8975 | 86.11% | 93.23% | ~12 ms |
| **LightGBM Classifier (Selected)** | **0.9985** | **0.9314** | **88.19%** | **95.49%** | **~2 ms** |
| Isolation Forest (Unsupervised) | 0.9777 | 0.6655 | 65.97% | 71.43% | ~5 ms |

### Tuned Business Impact Metrics
Using the tuned cost matrix balancing fraud loss against customer friction:
- **Fraud Value Prevented:** **~94.0%** of total attempted fraudulent funds stopped.
- **Legitimate Customer Friction:** Only **~1.5%** of normal transactions subjected to verification.
- **Scenarios Evaluated:** Account Takeover (98.4% recall), Money-Mule Networks (96.8% recall), Structuring Smurfing (95.1% recall), Rogue Agents (92.3% recall), and Scam Victims (81.2% recall).

---

## 13. Conclusion & Roadmap

### Conclusion
UpayShield bridges the critical gap between raw machine learning detection and operational compliance. By replacing black-box fraud scores with an auditable **three-part case narrative (What happened / Why risky / What next)**, network graph topology, and pre-transaction customer nudges in Bengali, UpayShield creates a safer, more transparent mobile money ecosystem for Bangladesh.

### Roadmap
- [x] **v1.0:** Time-split ML feature pipeline, LightGBM model, cost-tuned decision policy.
- [x] **v1.1:** FastAPI central backend, NetworkX graph engine, agent peer profiling, and grounded AI assistant.
- [x] **v1.2:** Unified static frontend hosting, Bengali/English warning demo, and automated test suite.
- [ ] **v2.0 (Planned):** Semi-supervised graph neural networks (PyTorch Geometric) for deep mule ring discovery.
- [ ] **v2.1 (Planned):** Biometric and behavioral device telemetry SDK integration for mobile clients.
- [ ] **v2.2 (Planned):** Automated SAR (Suspicious Activity Report) PDF export formatted to Bangladesh Financial Intelligence Unit (BFIU) regulatory standards.

---

## 14. Team & Contributions

Developed with pride for the **Upay AI Dev Fest (Track 01: Trust & Risk Intelligence)**:

| Name | Role | Responsibilities & Ownership | GitHub |
|---|---|---|---|
| **Ridwan Siddque** | **ML & Data Lead** | Synthetic dataset generation, 23 point-in-time features, LightGBM/Isolation Forest training, SHAP explainability, model evaluation, and feedback calibration. | [@Ridwan-Rythm](https://github.com/Ridwan-Rythm) |
| **Sakib Hasan** | **Backend & Intelligence Lead** | FastAPI architecture, NetworkX graph intelligence, agent peer-group modeling, decision policy engine, grounded AI investigation assistant, and test suites. | [@sakib-hsn](https://github.com/sakib.hsn44) |
| **Aritro Das** | **Frontend & Product Lead** | Analyst dashboard UI, interactive graph explorer, case investigation view, bilingual English/Bangla warning demo, and design system. | [@aritrodas](https://github.com/aritrodas) |

---
*License: MIT. Developed for Upay AI Dev Fest 2026.*
