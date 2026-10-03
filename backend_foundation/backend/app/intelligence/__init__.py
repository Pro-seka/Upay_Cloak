"""PART 2 REPLACES THIS FILE.  Public entry point: `build_intelligence(settings) -> IntelligenceServices`.

Until Part 2 lands, Part 1 runs against null stubs (valid-but-empty outputs).
"""
from __future__ import annotations

from backend.app.config import Settings
from backend.app.contracts.interfaces import IntelligenceServices
from backend.app.stubs import (
    NullAgentRiskService,
    NullAssistant,
    NullEvidenceBuilder,
    NullGraphService,
    NullReportExporter,
)


def build_intelligence(settings: Settings) -> IntelligenceServices:
    return IntelligenceServices(
        graph=NullGraphService(), agents=NullAgentRiskService(), evidence=NullEvidenceBuilder(),
        assistant=NullAssistant(), reports=NullReportExporter(), live=False)
