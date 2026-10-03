"""Contract smoke tests. Both parts must keep these green (run: pytest tests/backend/test_contracts.py)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.contracts import evidence_ids as eid
from backend.app.contracts.interfaces import (
    AgentRiskService,
    EvidenceBuilder,
    GraphService,
    IntelligenceServices,
    InvestigationAssistant,
    ReportExporter,
)
from backend.app.contracts.schemas import (
    ACTION_SEVERITY,
    ALERT_TYPE_PRECEDENCE,
    SEVERITY_TO_LEVEL,
    TAG_TO_ALERT_TYPE,
    Action,
    AlertType,
    Case,
    Decision,
    MLScore,
    PolicyHit,
    RiskLevel,
    ScoredTransaction,
    Transaction,
)
from backend.app.contracts.schemas_core import ConfigInfo, ScoreRequest, StreamEvent
from backend.app.contracts.schemas_intel import EvidenceBundle, EvidenceItem, NarrativeSentence
from backend.app.intelligence import build_intelligence

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "reports" / "sample_cases.json"


def test_ml_output_parses_into_contract():
    """reports/sample_cases.json is real RiskEngine.score_frame() output."""
    if not SAMPLE.exists():
        pytest.skip("run `python -m ml.train` first")
    cases = json.loads(SAMPLE.read_text())
    assert cases
    for raw in cases:
        m = MLScore.model_validate(raw)
        assert m.what_next.action in Action


def test_severity_and_levels_cover_all_actions():
    assert set(ACTION_SEVERITY) == set(Action)
    assert set(SEVERITY_TO_LEVEL.values()) == set(RiskLevel)
    assert ACTION_SEVERITY[Action.ALLOW] < ACTION_SEVERITY[Action.OTP_STEP_UP] < ACTION_SEVERITY[Action.HOLD]
    assert ACTION_SEVERITY[Action.HOLD] < ACTION_SEVERITY[Action.BLOCK] < ACTION_SEVERITY[Action.FREEZE_WALLET]


def test_alert_type_mapping_is_total_over_ml_tags():
    for tag in ("account_takeover", "scam_victim", "mule_passthrough", "mule_network", "structuring", "agent_risk"):
        assert TAG_TO_ALERT_TYPE[tag] in ALERT_TYPE_PRECEDENCE
    assert set(ALERT_TYPE_PRECEDENCE) == set(AlertType)


def _scored() -> ScoredTransaction:
    raw = {
        "txn_id": "TXN-0000001", "user_id": "U00001",
        "what_happened": {"type": "TRANSFER", "amount_bdt": 4190.0, "recipient": "MU0-0", "device": "DX1",
                          "location": "Sylhet", "time": "2026-01-26 01:40:13"},
        "why_risky": {"risk_score": 0.99, "model_score": 0.99, "anomaly_score": 0.99},
        "what_next": {"action": "block", "risk_level": "critical", "base_action": "block"},
    }
    return ScoredTransaction.model_validate(raw)


def test_case_and_stream_event_roundtrip():
    s = _scored()
    now = datetime.now(timezone.utc)
    case = Case(case_id="CASE-0000001", created_at=now, updated_at=now, scored=s,
                alert_type=AlertType.SCAM, priority=4148.1)
    again = Case.model_validate_json(case.model_dump_json())
    assert again.scored.what_next.action == Action.BLOCK
    ev = StreamEvent(seq=1, sim_time="2026-01-26T01:40:13", kind="alert", scored=s, alert_type=AlertType.SCAM)
    assert StreamEvent.model_validate_json(ev.model_dump_json()).seq == 1


def test_decision_is_constructible():
    d = Decision(action=Action.FREEZE_WALLET, risk_level=RiskLevel.CRITICAL, base_action=Action.BLOCK,
                 policy_trace=[PolicyHit(policy_id="POL_RING_FREEZE", text="x", raises_to=Action.FREEZE_WALLET)])
    assert d.requires_analyst is False


def test_transaction_rejects_bad_amounts():
    with pytest.raises(ValidationError):
        Transaction(ts=datetime.now(), user_id="u", type="TRANSFER", amount=0, recipient_id="r",
                    device_id="d", location="Dhaka", balance_before=1)


def test_evidence_ids_roundtrip_and_regex():
    assert eid.txn("TXN-0062110") == "TXN-0062110" == eid.txn("0062110")
    assert eid.wallet("MU0-0") == "WALLET-MU0-0"
    text = (f"Sent to {eid.wallet('MU0-0')} via {eid.device('DX932403')}, "
            f"see {eid.txn('TXN-0062110')}. Also {eid.rule('ATO_01')}.")
    assert eid.extract_ids(text) == ["WALLET-MU0-0", "DEVICE-DX932403", "TXN-0062110", "RULE-ATO_01"]
    assert eid.extract_ids("no ids here, TXN alone, WALLET- nothing") == []


def test_bundle_helpers():
    b = EvidenceBundle(case_id="CASE-1", generated_at=datetime.now(timezone.utc),
                       recommended_action=Action.HOLD, what_happened=["TXN-1"], why_risky=[], what_next=[],
                       items=[EvidenceItem(id="TXN-1", kind="TXN", label="x", facts={"amount_bdt": 5.0})])
    assert b.ids() == {"TXN-1"} and b.index()["TXN-1"].facts["amount_bdt"] == 5.0


def test_narrative_sentence_requires_evidence():
    with pytest.raises(ValidationError):
        NarrativeSentence(text="unsupported claim", evidence_ids=[])


def test_null_stubs_satisfy_protocols():
    intel = build_intelligence(Settings())
    assert isinstance(intel, IntelligenceServices) and intel.live is False
    assert isinstance(intel.graph, GraphService)
    assert isinstance(intel.agents, AgentRiskService)
    assert isinstance(intel.evidence, EvidenceBuilder)
    assert isinstance(intel.assistant, InvestigationAssistant)
    assert isinstance(intel.reports, ReportExporter)


def test_settings_boot_with_zero_config():
    s = Settings()
    assert s.llm_provider == "none" and s.llm_timeout_s > 0
    ConfigInfo(contract_version="1.0", app_version="0", model_loaded=False, intelligence="stubs",
               thresholds_model=[0.2, 0.25, 0.3], thresholds_active=[0.2, 0.25, 0.3], thresholds_version="model",
               risk_levels={}, actions=list(Action), languages=["en", "bn"], llm_provider="none", llm_enabled=False)
    ScoreRequest(txn_id="TXN-1")
