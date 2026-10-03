"""Intelligence layer factory for UpayShield (Part 2 implementation).
Builds and wires graph analytics, agent risk profiling, structured evidence generation,
grounded AI assistant, and compliance report exporting.
"""
from __future__ import annotations

from backend.app.config import Settings
from backend.app.contracts.interfaces import IntelligenceServices
from backend.app.intelligence.agents import PeerAgentRiskService
from backend.app.intelligence.assistant import GroundedInvestigationAssistant
from backend.app.intelligence.evidence import StructuredEvidenceBuilder
from backend.app.intelligence.graph import NetworkXGraphService
from backend.app.intelligence.reports import MarkdownReportExporter


def build_intelligence(settings: Settings, live: bool = False) -> IntelligenceServices:
    """Build Part 2 intelligence services.
    Defaults to live=False for contract testing, live=True when started in production/main.
    """
    return IntelligenceServices(
        graph=NetworkXGraphService(),
        agents=PeerAgentRiskService(),
        evidence=StructuredEvidenceBuilder(),
        assistant=GroundedInvestigationAssistant(),
        reports=MarkdownReportExporter(),
        live=live,
    )
