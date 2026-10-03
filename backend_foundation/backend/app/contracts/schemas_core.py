"""Models owned by BACKEND PART 1 (core API & decisioning).

Part 1 may add/extend models here. Part 2 must treat this file as read-only.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from backend.app.contracts.schemas import (
    _M,
    Action,
    AlertType,
    Language,
    ScoredTransaction,
    Transaction,
    TxnType,
)


# ---------------------------------------------------------------- /config
class ConfigInfo(_M):
    contract_version: str
    app_version: str
    model_loaded: bool
    intelligence: Literal["live", "stubs"]
    thresholds_model: list[float]            # [otp, hold, block] learned by ml.train
    thresholds_active: list[float]           # after analyst-feedback adaptation
    thresholds_version: str                  # "model" | "adaptive-v3"
    risk_levels: dict[str, str]              # level -> human description of its action tier
    actions: list[Action]
    languages: list[Language]
    llm_provider: str
    llm_enabled: bool
    dataset: dict[str, str | int | float | None] = {}   # rows, start, end


# ---------------------------------------------------------------- scoring
class ScoreRequest(_M):
    """Either replay an existing transaction by id, or score a raw one."""
    txn_id: str | None = None
    transaction: Transaction | None = None
    commit: bool = False             # raw txn only: persist into history + case store


class ScoreResponse(_M):
    scored: ScoredTransaction
    latency_ms: float
    feature_mode: Literal["replay", "live"]


# ---------------------------------------------------------------- customer warning (pre-send nudge)
class WarningRequest(_M):
    user_id: str
    recipient_id: str
    amount: float = Field(gt=0)
    type: TxnType = "TRANSFER"
    language: Language = "en"
    device_id: str | None = None
    location: str | None = None


class WarningResult(_M):
    show_warning: bool
    severity: Literal["none", "info", "caution", "danger"]
    language: Language
    headline: str
    body: str
    reasons: list[str] = []          # localized, customer-safe sentences (never raw scores)
    suggested_action: Action = Action.ALLOW
    evidence_ids: list[str] = []


# ---------------------------------------------------------------- dashboard
class TimeBucket(_M):
    ts: str
    count: int


class ReasonCount(_M):
    text: str
    count: int


class KpiSummary(_M):
    sim_time: str | None = None
    total_txns_scored: int
    alerts: int
    open_cases: int
    blocked_value_bdt: float
    alerts_by_type: dict[str, int] = {}
    risk_mix: dict[str, int] = {}            # low/medium/high/critical -> count
    alerts_over_time: list[TimeBucket] = []
    top_reasons: list[ReasonCount] = []


class FeedbackStats(_M):
    confirmed: int = 0
    dismissed: int = 0
    precision_confirmed: float | None = None     # confirmed / (confirmed + dismissed)
    by_alert_type: dict[str, dict[str, int]] = {}


class ImpactMetrics(_M):
    """Business-impact panel. Model numbers come from reports/metrics.json; feedback is live."""
    n_test: int
    roc_auc: float
    pr_auc: float
    precision_at_1pct: float
    recall_at_1pct: float
    fraud_value_total_bdt: float
    fraud_value_prevented_bdt: float
    pct_fraud_value_prevented: float
    legit_txns_with_friction_pct: float
    alert_rate_pct: float
    net_benefit_bdt: float
    analyst_queue_size: int
    actions: dict[str, int] = {}
    scenario_recall: dict[str, float] = {}
    notes: str = ""
    analyst_feedback: FeedbackStats = FeedbackStats()


# ---------------------------------------------------------------- stream
class StreamState(_M):
    running: bool
    mode: Literal["replay", "demo"]
    interval_ms: int
    seq: int
    sim_time: str | None = None


class StreamControl(_M):
    action: Literal["start", "pause", "resume", "reset"]
    mode: Literal["replay", "demo"] | None = None
    interval_ms: int | None = Field(None, ge=100, le=60000)


class StreamEvent(_M):
    seq: int
    sim_time: str
    kind: Literal["txn", "alert"]
    scored: ScoredTransaction
    alert_type: AlertType | None = None
