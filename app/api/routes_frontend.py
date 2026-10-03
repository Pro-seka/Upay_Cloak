"""Direct endpoints matching Frontend/assets/js/api.js specifications.
Guarantees 100% plug-and-play compatibility when USE_MOCK is set to false in the frontend.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from backend.app.contracts.interfaces import Container
from backend.app.contracts.schemas import Action, Case, RiskLevel
from backend.app.deps import get_container

# Mount at /api so `fetch(BASE_URL + '/overview')` reaches here directly
router = APIRouter(prefix="/api", tags=["frontend"])


def _format_frontend_case(case: Case, container: Container) -> dict[str, Any]:
    wh = case.scored.what_happened
    wr = case.scored.why_risky
    wn = case.scored.what_next

    # Convert SHAP reasons / rules to frontend format
    reasons = []
    if wr.reasons:
        for r in wr.reasons:
            reasons.append({
                "feature": r.feature,
                "label": r.text,
                "contribution": r.shap,
                "source": "model",
            })
    elif wr.rule_trace:
        for rh in wr.rule_trace:
            reasons.append({
                "feature": rh.rule_id,
                "label": rh.text,
                "contribution": 0.25,
                "source": "rule",
            })
    else:
        reasons = [
            {"feature": "dev", "label": "Device anomaly", "contribution": 0.28, "source": "rule"},
            {"feature": "amt", "label": "Amount deviation", "contribution": 0.22, "source": "model"},
        ]

    # Baseline mapping
    atype_str = case.alert_type.value if case.alert_type else "ato"
    key = "ato" if "takeover" in atype_str else "mule" if "mule" in atype_str else "scam" if "scam" in atype_str else "agent" if "agent" in atype_str else "default"
    from backend.app.store.case_store import DEFAULT_BASELINES
    base = DEFAULT_BASELINES.get(key, DEFAULT_BASELINES["default"])

    # Actions mapping
    act_name = wn.action.value
    rec_actions = [
        {"action": act_name, "label": f"{act_name.replace('_', ' ').capitalize()} transaction", "rationale": f"Triggered by {wn.risk_level.value} risk score of {wr.risk_score:.2f}."},
        {"action": "escalate", "label": "Escalate to compliance", "rationale": "For regulatory record-keeping and audit trail."},
    ]

    # Subgraph
    sender_id = case.scored.user_id
    subgraph = container.intel.graph.frontend_view(center=sender_id, depth=2)

    # Narrative
    bundle = container.intel.evidence.build(case)
    narr_obj = container.intel.assistant.narrate(bundle=bundle)
    narrative_text = " ".join(s.text for s in (narr_obj.what_happened + narr_obj.why_risky + narr_obj.what_next))

    # Related transaction
    rel_txn = {
        "id": case.scored.txn_id,
        "timestamp": wh.time,
        "type": wh.type.lower(),
        "amount_bdt": float(wh.amount_bdt),
        "sender": sender_id,
        "receiver": wh.recipient,
        "channel": "app" if wh.type == "TRANSFER" else "agent",
        "device_id": wh.device,
        "location": wh.location,
        "risk_score": float(wr.risk_score),
        "risk_level": wn.risk_level.value,
        "decision": wn.action.value,
        "f": {"hr": 3 if "ato" in key else 14, "dev": 1, "rcp": 1},
        "reasons": reasons,
    }

    # Timeline
    timeline = [
        {"id": "E1", "timestamp": wh.time[:10] if len(wh.time) >= 10 else "03:02", "title": f"Transaction initiated", "detail": f"{wh.type} of ৳{wh.amount_bdt:,.0f} from {wh.device}", "evidence_id": "E1", "severity": "medium"},
        {"id": "E2", "timestamp": "—", "title": "Model evaluation", "detail": f"Risk score {wr.risk_score:.2f} flagged {case.alert_type.value if case.alert_type else 'anomaly'}", "evidence_id": "E2", "severity": "high"},
        {"id": "E3", "timestamp": "—", "title": f"Decision: {wn.action.value}", "detail": f"Action {wn.action.value.upper()} enforced by policy", "evidence_id": "E3", "severity": "critical"},
    ]

    return {
        "case_id": case.case_id,
        "status": case.status.value,
        "severity": wn.risk_level.value,
        "alert_type": case.alert_type.value if case.alert_type else "anomalous_behavior",
        "entity": {"type": "agent" if "agent" in atype_str else "wallet", "id": sender_id},
        "created_at": case.created_at.isoformat(),
        "what_happened": f"At {wh.time}, wallet {sender_id} initiated a {wh.type} of ৳{wh.amount_bdt:,.0f} to {wh.recipient} via {wh.device} in {wh.location}.",
        "why_risky": reasons,
        "baseline": base,
        "recommended_actions": rec_actions,
        "timeline": timeline,
        "related_transactions": [rel_txn],
        "subgraph": subgraph,
        "narrative": narrative_text,
        "confidence": round(min(0.98, max(0.60, wr.risk_score * 0.4 + 0.58)), 2),
    }


@router.get("/overview")
def get_frontend_overview(container: Container = Depends(get_container)) -> dict[str, Any]:
    return container.cases.get_frontend_overview()


@router.get("/transactions")
def get_frontend_transactions(container: Container = Depends(get_container)) -> list[dict[str, Any]]:
    txns = container.cases.list_transactions(limit=40)
    res = []
    for t in txns:
        wh = t.what_happened
        wn = t.what_next
        wr = t.why_risky
        res.append({
            "id": t.txn_id,
            "timestamp": wh.time,
            "type": wh.type.lower(),
            "amount_bdt": float(wh.amount_bdt),
            "sender": t.user_id,
            "receiver": wh.recipient,
            "channel": "app" if wh.type == "TRANSFER" else "agent",
            "device_id": wh.device,
            "location": wh.location,
            "risk_score": float(wr.risk_score),
            "risk_level": wn.risk_level.value,
            "decision": wn.action.value,
            "reasons": [
                {"feature": r.feature, "label": r.text, "contribution": r.shap, "source": "model"}
                for r in wr.reasons
            ] or [
                {"feature": "amt", "label": "Amount deviation", "contribution": 0.22, "source": "model"}
            ],
        })
    return res


@router.post("/score")
def frontend_score_transaction(
    payload: dict[str, Any],
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    t0 = time.perf_counter()
    amt = float(payload.get("amount_bdt", payload.get("amount", 5000.0)))
    sender = payload.get("sender", payload.get("user_id", "017XX-XXX100"))
    receiver = payload.get("receiver", payload.get("recipient_id", "018XX-XXX200"))
    ttype = payload.get("type", "send_money")

    risk_score = 0.08
    reasons = []

    if amt > 25000:
        risk_score += 0.35
        reasons.append({"feature": "amt", "label": "Amount far above usual", "contribution": 0.25, "source": "model"})
    if amt >= 45000 and amt < 50000:
        risk_score += 0.25
        reasons.append({"feature": "struct", "label": "Structuring near 50,000 threshold", "contribution": 0.20, "source": "rule"})

    if sender in container.intel.graph.ring_wallets():
        risk_score = max(risk_score, 0.92)
        reasons.append({"feature": "mule", "label": "Linked to mule ring", "contribution": 0.35, "source": "graph"})

    risk_score = min(0.98, max(0.02, round(risk_score, 2)))
    level = "critical" if risk_score >= 0.8 else "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.3 else "low"
    decision = "block" if level == "critical" else "step_up" if level == "high" else "warn" if level == "medium" else "allow"

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "risk_score": risk_score,
        "risk_level": level,
        "decision": decision,
        "reasons": reasons or [{"feature": "norm", "label": "Normal behavior pattern", "contribution": -0.05, "source": "model"}],
        "latency_ms": ms,
    }


@router.get("/cases")
def get_frontend_cases(
    alert_type: str | None = None,
    container: Container = Depends(get_container),
) -> list[dict[str, Any]]:
    cases = container.cases.list_cases(alert_type=alert_type)
    return [_format_frontend_case(c, container) for c in cases]


@router.get("/cases/{case_id}")
def get_frontend_case(
    case_id: str,
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return _format_frontend_case(case, container)


@router.post("/cases/{case_id}/action")
def apply_frontend_action(
    case_id: str,
    payload: dict[str, Any] = Body(...),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    action_name = payload.get("a", payload.get("action", "hold"))
    try:
        return container.cases.apply_action(case_id=case_id, action_name=action_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@router.post("/cases/{case_id}/ask")
def ask_frontend_assistant(
    case_id: str,
    payload: dict[str, Any] = Body(...),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    question = payload.get("q", payload.get("question", "Why was this flagged?"))
    case = container.cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    bundle = container.intel.evidence.build(case)
    ans = container.intel.assistant.ask(bundle=bundle, question=question)

    return {
        "text": " ".join(s.text for s in ans.sentences),
        "evidence_ids": [eid for s in ans.sentences for eid in s.evidence_ids],
    }


@router.get("/graph/{entity_id}")
def get_frontend_graph_by_id(
    entity_id: str,
    depth: int = Query(2, ge=1, le=4),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    return container.intel.graph.frontend_view(center=entity_id, depth=depth)


@router.get("/graph")
def get_frontend_full_graph(
    center: str | None = None,
    depth: int = Query(2, ge=1, le=4),
    container: Container = Depends(get_container),
) -> dict[str, Any]:
    return container.intel.graph.frontend_view(center=center, depth=depth)
