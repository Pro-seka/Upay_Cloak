# UpayShield Backend: API & Integration Contract (v1.0)

This file is the **single source of truth** for how Backend Part 1 (core API & decisioning) and
Backend Part 2 (intelligence layer) fit together. The Python versions of these contracts live in
`backend/app/contracts/` and are **frozen** while the two parts are built in parallel.

> Every response answers the judges' three questions:
> **what_happened** (facts) · **why_risky** (score, SHAP, rules, graph, agent) · **what_next** (decision).

---

## 1. Conventions

| Topic | Rule |
|---|---|
| Base path | `/api/v1` (except `GET /health`). OpenAPI at `/docs`. |
| Format | JSON, UTF-8 (Bangla text must round-trip). CSV/Markdown/PDF only where stated. |
| Timestamps | Replayed transactions keep the dataset's naive ISO format (`2026-01-26T01:40:13`). Server-generated timestamps (`created_at`, `generated_at`) are timezone-aware UTC (`...Z`). |
| Money | `amount_bdt` / `*_bdt` fields are BDT (float). |
| Errors | Always `{"error": {"code": str, "message": str, "details": {...}|null}}` (`ErrorResponse`). Codes are `snake_case`: `case_not_found`, `txn_not_found`, `ring_not_found`, `agent_not_found`, `validation_error`, `model_not_loaded`, `report_missing`, `internal_error`. |
| Status codes | 200 OK · 201 created · 404 not found · 422 validation · 503 only when the ML artifacts or reports are missing (`model_not_loaded`, `report_missing`). **The LLM path never returns 5xx**: it falls back to the template narrative. |
| Pagination | `?limit=50&cursor=<opaque>` → `Page{items,total,next_cursor}`. |
| CORS | Allow all origins by default (static frontend opened from file/other port). |
| Idempotency | `case_id = "CASE-" + txn_id suffix` (e.g. `TXN-0062110` → `CASE-0062110`). Re-scoring the same txn never duplicates a case. |
| Labels | `is_fraud` / `scenario` are **dataset labels**. They may be used for evaluation and the freeze what-if only, never as model/graph/agent inputs, never in customer-facing output. |

---

## 2. Endpoints and owners

### Part 1 (files: `backend/app/main.py`, `api/routes_core.py`, `services/`, `store/`, `i18n/`)

| Method & path | Request | Response |
|---|---|---|
| `GET /health` | | `{"status":"ok","model_loaded":bool}` |
| `GET /api/v1/config` | | `ConfigInfo` |
| `POST /api/v1/score` | `ScoreRequest` (`txn_id` **or** `transaction`; `?commit`) | `ScoreResponse` |
| `GET /api/v1/transactions` | `limit, offset, risk_level, user_id, since` | `Page[ScoredTransaction]` |
| `GET /api/v1/cases` | `status, alert_type, risk_level, min_amount, sort=priority\|time, limit, cursor` | `Page[Case]` |
| `GET /api/v1/cases/{case_id}` | | `Case` (graph + agent evidence already attached in `scored.why_risky`) |
| `POST /api/v1/cases/{case_id}/feedback` | `{verdict: confirm\|dismiss, analyst?, note?}` | `Case` |
| `GET /api/v1/feedback/export` | | `text/csv` (labels for retraining) |
| `POST /api/v1/admin/thresholds/reset` | | `ConfigInfo` |
| `GET /api/v1/kpis` | | `KpiSummary` |
| `GET /api/v1/metrics` | | `ImpactMetrics` |
| `GET /api/v1/stream` | `?mode=replay\|demo&interval_ms=` | SSE (see §5) |
| `GET /api/v1/stream/state` | | `StreamState` |
| `POST /api/v1/stream/control` | `StreamControl` | `StreamState` |
| `POST /api/v1/warning/check` | `WarningRequest` | `WarningResult` |

### Part 2 (files: `backend/app/api/routes_intel.py`, `backend/app/intelligence/**`)

| Method & path | Request | Response |
|---|---|---|
| `GET /api/v1/graph` | `center` (wallet/device/agent id), `depth=2`, `max_nodes=120`, `as_of?` | `GraphView` |
| `GET /api/v1/graph/rings` | `min_score=0.5` | `list[RingSummary]` |
| `GET /api/v1/graph/rings/{ring_id}` | | `RingSummary` |
| `POST /api/v1/graph/simulate-freeze` | `FreezeRequest` | `FreezeImpact` |
| `GET /api/v1/agents` | `top_n=10` | `list[AgentRisk]` (leaderboard) |
| `GET /api/v1/agents/{agent_id}` | | `AgentRisk` |
| `GET /api/v1/cases/{case_id}/evidence` | | `EvidenceBundle` |
| `POST /api/v1/cases/{case_id}/narrative` | `NarrativeRequest` | `Narrative` |
| `POST /api/v1/cases/{case_id}/ask` | `AskRequest` | `Answer` |
| `GET /api/v1/cases/{case_id}/report` | `format=md\|pdf`, `language=en\|bn` | `text/markdown` or `application/pdf` |

Route handlers in `routes_intel.py` get their dependencies with `Depends(get_container)` (`backend/app/deps.py`).

---

## 3. Decision semantics (owned by Part 1, relied on by everyone)

**Base score and tiers.** `ml.score.RiskEngine` returns a blended `risk_score` in `[0,1]` and a base
action using thresholds `[t_otp, t_hold, t_block]` learned by `ml.train` (currently `[0.20, 0.25, 0.30]`
on the blended score; most legitimate traffic sits below 0.20).

| Action | Severity | `risk_level` |
|---|---|---|
| `allow` | 0 | `low` |
| `otp_step_up` | 1 | `medium` |
| `hold` | 2 | `high` |
| `escalate` (hold + senior/compliance review) | 3 | `high` |
| `block` | 4 | `critical` |
| `freeze_wallet` | 5 | `critical` |

**Policies may only raise severity.** `Decision.base_action` keeps what the model alone would do;
`Decision.action` is `max(base, policies)`; every raise is recorded in `policy_trace`.

**`alert_type`**: ML tags map through `TAG_TO_ALERT_TYPE`; with several tags the first in
`ALERT_TYPE_PRECEDENCE` wins; no tag but flagged → `anomaly`; graph/agent policies may set
`mule_network` / `agent_anomaly`.

**Cases and priority.** A case exists iff `action != allow`. `priority = risk_score * amount_bdt`.
Default queue order = priority desc.

> **Frontend note:** `settings.html` currently documents fixed bands (0.30/0.65/0.80). The real tiers
> come from `GET /api/v1/config` (`thresholds_model`, `thresholds_active`, `risk_levels`). The UI should
> render those instead of hard-coding numbers.

---

## 4. Evidence IDs

Built **only** through `backend/app/contracts/evidence_ids.py`:

`TXN-0062110` · `WALLET-U01555` · `DEVICE-DX932403` · `AGENT-A012` · `RING-3fa9c1` · `RULE-ATO_01` ·
`FACTOR-amount_z` · `POLICY-POL_RING_FREEZE` · `METRIC-<name>`

Narrative sentences must cite ≥1 id that exists in the case's `EvidenceBundle`; any number in a sentence
must appear in the `facts` of a cited item.

---

## 5. Stream (SSE)

`GET /api/v1/stream` → `text/event-stream`.

```
id: 1042
event: txn            # every transaction   (data = StreamEvent JSON, kind="txn")
event: alert          # action != allow     (data = StreamEvent JSON, kind="alert")
event: heartbeat      # every 15 s          (data = {"sim_time": "..."})
```

* One global simulation clock shared by all clients; `Last-Event-ID` resumes.
* `mode=replay`: chronological, real mix (alerts ≈ 2 % of events).
* `mode=demo`: still chronological, but guarantees an alert at least every `stream_demo_alert_every`
  events by drawing from pre-scored alerts. **Label this honestly** in `StreamState.mode`.

---

## 6. Cross-part flow

```
startup (Part 1):  load artifacts → build_features → RiskEngine.score_frame (cached)
                   intel = build_intelligence(settings)          # Part 2 factory
                   intel.graph.build(df); intel.agents.build(df)
per transaction:   ML score → graph.signals(txn_id) + agents.agent_risk(agent_id, as_of=ts)
                   → decision engine (policies) → case store          # Part 1
case drill-down:   GET /cases/{id}/evidence → EvidenceBuilder.build(case)    # Part 2
                   POST /cases/{id}/narrative → InvestigationAssistant.narrate(bundle)
```

`GraphService` and `AgentRiskService` are **point-in-time**: no information from after `as_of`.
