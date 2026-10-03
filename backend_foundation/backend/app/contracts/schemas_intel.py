"""Models owned by BACKEND PART 2 (intelligence layer).

Part 2 may add/extend models here. Part 1 must treat this file as read-only.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from backend.app.contracts.schemas import _M, Action, Language, Scalar


# ---------------------------------------------------------------- graph
class GraphNode(_M):
    id: str                                   # wallet / device / agent id as it appears in the data
    kind: Literal["wallet", "device", "agent"]
    label: str
    risk: float = Field(0.0, ge=0, le=1)      # graph heuristic, or max ML risk when attach_scores() was called
    ring_id: str | None = None
    community_id: int | None = None
    flags: list[str] = []
    evidence_id: str                          # e.g. WALLET-U01555 / DEVICE-D01 / AGENT-A012


class GraphEdge(_M):
    source: str
    target: str
    kind: Literal["transfer", "cash_out", "cash_in", "uses_device"]
    amount_bdt: float = 0.0
    count: int = 1
    first_ts: str | None = None
    last_ts: str | None = None
    txn_ids: list[str] = []                   # capped (<= 10) sample for drill-down


class GraphView(_M):
    center: str | None = None
    as_of: datetime | None = None
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    stats: dict[str, float | int] = {}


class RingSummary(_M):
    ring_id: str
    wallet_ids: list[str]
    size: int
    ring_score: float = Field(ge=0, le=1)
    total_inflow_bdt: float
    total_outflow_bdt: float
    victim_wallets: int                       # distinct upstream senders into the ring
    shared_devices: list[str] = []
    cashout_agents: list[str] = []
    first_seen: str
    last_seen: str
    flags: list[str] = []
    evidence_ids: list[str] = []


class FreezeRequest(_M):
    wallet_ids: list[str] = Field(min_length=1, max_length=50)
    freeze_at: datetime | None = None         # default: ring first_seen + 30 min


class FreezeImpact(_M):
    """'What if we freeze these wallets?'  Uses dataset labels -> labels_used=True (demo only)."""
    frozen_wallets: list[str]
    freeze_at: datetime
    blocked_txn_count: int
    blocked_value_bdt: float
    fraud_value_stopped_bdt: float
    legit_value_blocked_bdt: float
    cashouts_prevented: int
    downstream_wallets_cut_off: int
    labels_used: bool = True


# ---------------------------------------------------------------- evidence
class EvidenceKind(str, Enum):
    TXN = "TXN"
    WALLET = "WALLET"
    DEVICE = "DEVICE"
    AGENT = "AGENT"
    RING = "RING"
    RULE = "RULE"
    FACTOR = "FACTOR"
    POLICY = "POLICY"
    METRIC = "METRIC"


class EvidenceItem(_M):
    id: str                                   # see evidence_ids.py
    kind: EvidenceKind
    label: str                                # short, factual, human-readable
    facts: dict[str, Scalar] = {}             # FLAT scalars only: the only numbers an LLM may quote


class TimelineEvent(_M):
    ts: str
    txn_id: str
    label: str
    amount_bdt: float | None = None
    focal: bool = False
    evidence_ids: list[str] = []


class EvidenceBundle(_M):
    case_id: str
    generated_at: datetime
    recommended_action: Action
    what_happened: list[str]                  # evidence ids answering Q1
    why_risky: list[str]                      # evidence ids answering Q2
    what_next: list[str]                      # evidence ids answering Q3
    items: list[EvidenceItem]
    timeline: list[TimelineEvent] = []

    def index(self) -> dict[str, EvidenceItem]:
        return {i.id: i for i in self.items}

    def ids(self) -> set[str]:
        return {i.id for i in self.items}


# ---------------------------------------------------------------- narrative
class NarrativeSentence(_M):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Narrative(_M):
    case_id: str
    language: Language
    source: Literal["llm", "template"]
    validated: bool                           # True = passed the grounding validator
    what_happened: list[NarrativeSentence]
    why_risky: list[NarrativeSentence]
    what_next: list[NarrativeSentence]
    recommended_action: Action
    model: str | None = None
    fallback_reason: str | None = None        # set when source == "template" because the LLM path failed


class AskRequest(_M):
    question: str = Field(min_length=3, max_length=500)
    language: Language = "en"


class Answer(_M):
    case_id: str
    question: str
    language: Language
    source: Literal["llm", "template"]
    validated: bool
    answerable: bool = True                   # False -> "the evidence does not say"
    sentences: list[NarrativeSentence]
    fallback_reason: str | None = None


class NarrativeRequest(_M):
    language: Language = "en"
    force_template: bool = False
