"""Null implementations of the Part 2 services: valid, empty answers, never raise.

Lets Part 1 run end-to-end before Part 2 is merged.  `build_intelligence()` in
`backend/app/intelligence/__init__.py` returns these until Part 2 replaces that file.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd

from backend.app.contracts.schemas import AgentRisk, Case, GraphSignals, Language, Transaction
from backend.app.contracts.schemas_intel import (
    Answer,
    EvidenceBundle,
    FreezeImpact,
    GraphView,
    Narrative,
    RingSummary,
)


class NullGraphService:
    def build(self, txns: pd.DataFrame) -> None: ...
    def ingest(self, txn: Transaction) -> None: ...
    def attach_scores(self, risk_by_txn: Mapping[str, float]) -> None: ...
    def signals(self, txn_id: str) -> GraphSignals: return GraphSignals(wallet_id="unknown")
    def wallet_signals(self, wallet_id: str, as_of: datetime | None = None) -> GraphSignals:
        return GraphSignals(wallet_id=wallet_id, as_of=as_of)
    def view(self, center: str, depth: int = 2, max_nodes: int = 120, as_of=None) -> GraphView:
        return GraphView(center=center, nodes=[], edges=[])
    def rings(self, min_score: float = 0.5, as_of=None) -> list[RingSummary]: return []
    def ring(self, ring_id: str) -> RingSummary: raise KeyError(ring_id)
    def ring_wallets(self) -> frozenset[str]: return frozenset()
    def simulate_freeze(self, wallet_ids, freeze_at=None) -> FreezeImpact:
        return FreezeImpact(frozen_wallets=wallet_ids, freeze_at=freeze_at or datetime.now(timezone.utc),
                            blocked_txn_count=0, blocked_value_bdt=0.0, fraud_value_stopped_bdt=0.0,
                            legit_value_blocked_bdt=0.0, cashouts_prevented=0, downstream_wallets_cut_off=0)


class NullAgentRiskService:
    def build(self, txns: pd.DataFrame) -> None: ...
    def ingest(self, txn: Transaction) -> None: ...
    def agent_risk(self, agent_id: str, as_of=None) -> AgentRisk: return AgentRisk(agent_id=agent_id)
    def leaderboard(self, top_n: int = 10, as_of=None) -> list[AgentRisk]: return []


class NullEvidenceBuilder:
    def build(self, case: Case) -> EvidenceBundle:
        return EvidenceBundle(case_id=case.case_id, generated_at=datetime.now(timezone.utc),
                              recommended_action=case.scored.what_next.action,
                              what_happened=[], why_risky=[], what_next=[], items=[])


class NullAssistant:
    def narrate(self, bundle: EvidenceBundle, language: Language = "en", force_template: bool = False) -> Narrative:
        return Narrative(case_id=bundle.case_id, language=language, source="template", validated=True,
                         what_happened=[], why_risky=[], what_next=[],
                         recommended_action=bundle.recommended_action, fallback_reason="intelligence stubs")
    def ask(self, bundle: EvidenceBundle, question: str, language: Language = "en") -> Answer:
        return Answer(case_id=bundle.case_id, question=question, language=language, source="template",
                      validated=True, answerable=False, sentences=[], fallback_reason="intelligence stubs")


class NullReportExporter:
    def to_markdown(self, case, bundle, narrative) -> str: return f"# Case {case.case_id}\n\n(stub report)\n"
    def to_pdf(self, case, bundle, narrative) -> bytes: return b""
