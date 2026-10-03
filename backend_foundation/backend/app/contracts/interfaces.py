"""Service interfaces = the seam between Part 1 and Part 2.

Part 1 CONSUMES `GraphService`, `AgentRiskService`, `EvidenceBuilder`, `InvestigationAssistant`,
`ReportExporter` (via `Container`).  Part 2 IMPLEMENTS them and exposes `build_intelligence()`.
Part 1 IMPLEMENTS `CaseProvider`; Part 2 consumes it (its routes + evidence builder).

All time-dependent methods are POINT-IN-TIME: they may only use data with ts <= as_of
(default: the transaction's own timestamp, or "all data" for rings()/view() when as_of is None).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from backend.app.contracts.schemas import AgentRisk, Case, GraphSignals, Language, ScoredTransaction, Transaction
from backend.app.contracts.schemas_intel import (
    Answer,
    EvidenceBundle,
    FreezeImpact,
    GraphView,
    Narrative,
    RingSummary,
)

# columns every DataFrame handed to build() is guaranteed to have (data/transactions.csv):
TXN_COLUMNS = ["txn_id", "ts", "user_id", "type", "amount", "recipient_id", "agent_id",
               "device_id", "location", "balance_before", "is_fraud", "scenario"]
# `is_fraud` and `scenario` are LABELS.  Algorithms must never read them (evaluation/what-if only).


@runtime_checkable
class GraphService(Protocol):
    def build(self, txns: pd.DataFrame) -> None:
        """Bulk build from all history (sorted by ts). Idempotent; replaces previous state."""

    def ingest(self, txn: Transaction) -> None:
        """Incrementally add one live transaction (used when /score?commit=true)."""

    def attach_scores(self, risk_by_txn: Mapping[str, float]) -> None:
        """Optional: lets graph nodes carry max ML risk. No-op is acceptable."""

    def signals(self, txn_id: str) -> GraphSignals:
        """Evidence for the SENDER wallet of txn_id, as of that txn's timestamp."""

    def wallet_signals(self, wallet_id: str, as_of: datetime | None = None) -> GraphSignals: ...

    def view(self, center: str, depth: int = 2, max_nodes: int = 120,
             as_of: datetime | None = None) -> GraphView: ...

    def rings(self, min_score: float = 0.5, as_of: datetime | None = None) -> list[RingSummary]: ...

    def ring(self, ring_id: str) -> RingSummary: ...

    def ring_wallets(self) -> frozenset[str]:
        """O(1)-lookup set of every wallet currently in a detected ring (for cheap policy checks)."""

    def simulate_freeze(self, wallet_ids: list[str], freeze_at: datetime | None = None) -> FreezeImpact: ...


@runtime_checkable
class AgentRiskService(Protocol):
    def build(self, txns: pd.DataFrame) -> None: ...

    def ingest(self, txn: Transaction) -> None: ...

    def agent_risk(self, agent_id: str, as_of: datetime | None = None) -> AgentRisk: ...

    def leaderboard(self, top_n: int = 10, as_of: datetime | None = None) -> list[AgentRisk]: ...


@runtime_checkable
class CaseProvider(Protocol):
    """Implemented by Part 1 (SQLite + in-memory frames). Part 2 only reads through this."""

    def get_case(self, case_id: str) -> Case | None: ...

    def get_scored(self, txn_id: str) -> ScoredTransaction | None: ...

    def recent_transactions(self, wallet_id: str, until: datetime, limit: int = 20) -> list[Transaction]:
        """Transactions where wallet_id is sender OR recipient, newest first, strictly before `until`."""

    def history_frame(self) -> pd.DataFrame:
        """Full transaction frame (TXN_COLUMNS), sorted by ts. Treat as read-only."""


@runtime_checkable
class EvidenceBuilder(Protocol):
    def build(self, case: Case) -> EvidenceBundle: ...


@runtime_checkable
class InvestigationAssistant(Protocol):
    def narrate(self, bundle: EvidenceBundle, language: Language = "en",
                force_template: bool = False) -> Narrative: ...

    def ask(self, bundle: EvidenceBundle, question: str, language: Language = "en") -> Answer: ...


@runtime_checkable
class ReportExporter(Protocol):
    def to_markdown(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> str: ...

    def to_pdf(self, case: Case, bundle: EvidenceBundle, narrative: Narrative) -> bytes: ...


@dataclass
class IntelligenceServices:
    """Everything Part 2 delivers. `live` is False for the null stubs."""
    graph: GraphService
    agents: AgentRiskService
    evidence: EvidenceBuilder
    assistant: InvestigationAssistant
    reports: ReportExporter
    live: bool = True


@dataclass
class Container:
    """Wired at startup by Part 1 (backend/app/main.py) and injected with `Depends(get_container)`."""
    cases: CaseProvider
    intel: IntelligenceServices
