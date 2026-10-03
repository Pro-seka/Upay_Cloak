"""PART 2 OWNS THIS FILE: graph / agents / evidence / narrative / ask / report endpoints.
Provides endpoints under `/api/v1` matching the API Contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from backend.app.contracts.interfaces import Container
from backend.app.contracts.schemas import Action, AgentRisk, Language
from backend.app.contracts.schemas_intel import (
    Answer,
    AskRequest,
    EvidenceBundle,
    FreezeImpact,
    FreezeRequest,
    GraphView,
    Narrative,
    NarrativeRequest,
    RingSummary,
)
from backend.app.deps import get_container

router = APIRouter(prefix="/api/v1", tags=["intelligence"])


@router.get("/graph", response_model=GraphView)
def get_graph_view(
    center: str = Query("019XX-XXX305", description="Center entity ID (wallet, device, or agent)"),
    depth: int = Query(2, ge=1, le=4),
    max_nodes: int = Query(120, ge=10, le=300),
    as_of: datetime | None = None,
    container: Container = Depends(get_container),
) -> GraphView:
    return container.intel.graph.view(center=center, depth=depth, max_nodes=max_nodes, as_of=as_of)


@router.get("/graph/rings", response_model=list[RingSummary])
def get_rings(
    min_score: float = Query(0.5, ge=0.0, le=1.0),
    as_of: datetime | None = None,
    container: Container = Depends(get_container),
) -> list[RingSummary]:
    return container.intel.graph.rings(min_score=min_score, as_of=as_of)


@router.get("/graph/rings/{ring_id}", response_model=RingSummary)
def get_ring_by_id(
    ring_id: str,
    container: Container = Depends(get_container),
) -> RingSummary:
    try:
        return container.intel.graph.ring(ring_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": {"code": "ring_not_found", "message": f"Ring {ring_id} not found"}})


@router.post("/graph/simulate-freeze", response_model=FreezeImpact)
def simulate_freeze(
    req: FreezeRequest,
    container: Container = Depends(get_container),
) -> FreezeImpact:
    return container.intel.graph.simulate_freeze(wallet_ids=req.wallet_ids, freeze_at=req.freeze_at)


@router.get("/agents", response_model=list[AgentRisk])
def get_agent_leaderboard(
    top_n: int = Query(10, ge=1, le=50),
    as_of: datetime | None = None,
    container: Container = Depends(get_container),
) -> list[AgentRisk]:
    return container.intel.agents.leaderboard(top_n=top_n, as_of=as_of)


@router.get("/agents/{agent_id}", response_model=AgentRisk)
def get_agent_risk(
    agent_id: str,
    as_of: datetime | None = None,
    container: Container = Depends(get_container),
) -> AgentRisk:
    return container.intel.agents.agent_risk(agent_id=agent_id, as_of=as_of)


@router.get("/cases/{case_id}/evidence", response_model=EvidenceBundle)
def get_case_evidence(
    case_id: str,
    container: Container = Depends(get_container),
) -> EvidenceBundle:
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"error": {"code": "case_not_found", "message": f"Case {case_id} not found"}})
    return container.intel.evidence.build(case)


@router.post("/cases/{case_id}/narrative", response_model=Narrative)
def generate_case_narrative(
    case_id: str,
    req: NarrativeRequest = NarrativeRequest(),
    container: Container = Depends(get_container),
) -> Narrative:
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"error": {"code": "case_not_found", "message": f"Case {case_id} not found"}})
    bundle = container.intel.evidence.build(case)
    return container.intel.assistant.narrate(bundle=bundle, language=req.language, force_template=req.force_template)


@router.post("/cases/{case_id}/ask", response_model=Answer)
def ask_case_assistant(
    case_id: str,
    req: AskRequest,
    container: Container = Depends(get_container),
) -> Answer:
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"error": {"code": "case_not_found", "message": f"Case {case_id} not found"}})
    bundle = container.intel.evidence.build(case)
    return container.intel.assistant.ask(bundle=bundle, question=req.question, language=req.language)


@router.get("/cases/{case_id}/report")
def export_case_report(
    case_id: str,
    format: str = Query("md", pattern="^(md|pdf)$"),
    language: Language = "en",
    container: Container = Depends(get_container),
) -> Response:
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail={"error": {"code": "case_not_found", "message": f"Case {case_id} not found"}})
    bundle = container.intel.evidence.build(case)
    narrative = container.intel.assistant.narrate(bundle=bundle, language=language)

    if format == "pdf":
        pdf_bytes = container.intel.reports.to_pdf(case, bundle, narrative)
        return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=report_{case_id}.pdf"})
    else:
        md_text = container.intel.reports.to_markdown(case, bundle, narrative)
        return Response(content=md_text, media_type="text/markdown", headers={"Content-Disposition": f"inline; filename=report_{case_id}.md"})
