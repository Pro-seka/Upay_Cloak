"""Transaction and Case store for UpayShield.
Implements CaseProvider protocol and provides persistence for cases, audit logs, and feedback.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import get_settings
from backend.app.contracts.evidence_ids import extract_ids, txn as eid_txn

def case_from_txn(txn_id: str) -> str:
    cleaned = txn_id[4:] if txn_id.startswith("TXN-") else txn_id
    return f"CASE-{cleaned}"
from backend.app.contracts.interfaces import TXN_COLUMNS, CaseProvider
from backend.app.contracts.schemas import (
    Action,
    AlertType,
    Case,
    CaseStatus,
    Decision,
    Feedback,
    RiskLevel,
    ScoredTransaction,
    Transaction,
    Verdict,
    WhatHappened,
    WhyRisky,
)
from backend.app.contracts.schemas_core import FeedbackStats, KpiSummary, ReasonCount, TimeBucket

# Demo baseline templates for frontend
DEFAULT_BASELINES = {
    "ato": {
        "usual_amount": "৳600",
        "usual_hours": "9 AM – 9 PM",
        "usual_devices": "1 known phone",
        "usual_location": "Dhaka",
    },
    "mule": {
        "usual_amount": "৳900",
        "usual_hours": "9 AM – 9 PM",
        "usual_devices": "1 known phone",
        "usual_location": "Dhaka",
    },
    "scam": {
        "usual_amount": "৳500",
        "usual_hours": "9 AM – 9 PM",
        "usual_devices": "1 known phone",
        "usual_location": "Sylhet",
    },
    "agent": {
        "usual_amount": "৳4,000",
        "usual_hours": "9 AM – 6 PM",
        "usual_devices": "1 POS terminal",
        "usual_location": "Khulna",
    },
    "default": {
        "usual_amount": "৳700",
        "usual_hours": "9 AM – 9 PM",
        "usual_devices": "1 known phone",
        "usual_location": "Dhaka",
    },
}


class CaseStore(CaseProvider):
    def __init__(self, txns_path: Path | None = None, reports_dir: Path | None = None):
        settings = get_settings()
        self.txns_path = txns_path or settings.data_path
        self.reports_dir = reports_dir or settings.reports_dir

        self._txns_df: pd.DataFrame = pd.DataFrame()
        self._txns_by_id: dict[str, Transaction] = {}
        self._scored_by_txn_id: dict[str, ScoredTransaction] = {}
        self._cases_by_id: dict[str, Case] = {}
        self._audit_logs: dict[str, list[dict[str, Any]]] = {}
        self._feedback: list[Feedback] = []

        # Load data
        self.load()

    def load(self) -> None:
        """Load transactions and pre-scored cases."""
        if self.txns_path.exists():
            df = pd.read_csv(self.txns_path)
            # Ensure columns
            for col in TXN_COLUMNS:
                if col not in df.columns:
                    if col == "is_fraud":
                        df[col] = 0
                    elif col == "scenario":
                        df[col] = "normal"
                    elif col == "agent_id":
                        df[col] = None
                    else:
                        df[col] = None
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                df = df.sort_values("ts").reset_index(drop=True)
            self._txns_df = df

            # Cache transactions for fast lookup (first 10,000 for memory sanity)
            sample_df = df.tail(10000) if len(df) > 10000 else df
            for _, r in sample_df.iterrows():
                try:
                    t = Transaction(
                        txn_id=str(r["txn_id"]),
                        ts=pd.to_datetime(r["ts"]).to_pydatetime(),
                        user_id=str(r["user_id"]),
                        type=str(r["type"]),
                        amount=float(r["amount"]),
                        recipient_id=str(r["recipient_id"]),
                        agent_id=str(r["agent_id"]) if pd.notna(r["agent_id"]) else None,
                        device_id=str(r["device_id"]),
                        location=str(r["location"]),
                        balance_before=float(r["balance_before"]) if pd.notna(r["balance_before"]) else 0.0,
                    )
                    self._txns_by_id[t.txn_id] = t
                except Exception:
                    continue

        # Load sample cases if available
        sample_path = self.reports_dir / "sample_cases.json"
        if sample_path.exists():
            try:
                raw_cases = json.loads(sample_path.read_text(encoding="utf-8"))
                for rc in raw_cases:
                    self._ingest_raw_case(rc)
            except Exception as e:
                print(f"Warning: could not parse sample_cases.json: {e}")

        # Seed the 5 core demo cases if not present
        self._seed_demo_cases()

    def _ingest_raw_case(self, rc: dict[str, Any]) -> None:
        try:
            tid = rc.get("txn_id", "TXN-0000")
            cid = case_from_txn(tid)
            wh = rc.get("what_happened", {})
            wr = rc.get("why_risky", {})
            wn = rc.get("what_next", {})

            what_happened = WhatHappened(
                type=wh.get("type", "TRANSFER"),
                amount_bdt=float(wh.get("amount_bdt", wh.get("amount", 0.0))),
                recipient=wh.get("recipient", wh.get("recipient_id", "unknown")),
                device=wh.get("device", wh.get("device_id", "unknown")),
                location=wh.get("location", "Dhaka"),
                time=str(wh.get("time", wh.get("ts", datetime.now(timezone.utc).isoformat()))),
                agent=wh.get("agent"),
            )

            risk_score = float(wr.get("risk_score", 0.5))
            model_score = float(wr.get("model_score", risk_score))
            anomaly_score = float(wr.get("anomaly_score", 0.2))

            action_val = wn.get("action", "hold")
            try:
                act = Action(action_val)
            except ValueError:
                act = Action.HOLD

            risk_lvl = RiskLevel.HIGH if risk_score >= 0.65 else RiskLevel.MEDIUM

            decision = Decision(
                action=act,
                risk_level=risk_lvl,
                base_action=act,
                priority=round(risk_score * what_happened.amount_bdt, 2),
            )

            why_risky = WhyRisky(
                risk_score=risk_score,
                model_score=model_score,
                anomaly_score=anomaly_score,
                tags=wr.get("tags", []),
            )

            scored = ScoredTransaction(
                txn_id=tid,
                user_id=rc.get("user_id", "U001"),
                case_id=cid,
                what_happened=what_happened,
                why_risky=why_risky,
                what_next=decision,
            )

            case = Case(
                case_id=cid,
                status=CaseStatus.OPEN,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                alert_type=AlertType.ACCOUNT_TAKEOVER if "ato" in str(wr).lower() else AlertType.ANOMALY,
                priority=decision.priority,
                scored=scored,
            )

            self._scored_by_txn_id[tid] = scored
            self._cases_by_id[cid] = case
        except Exception as e:
            print(f"Error ingesting case: {e}")

    def _seed_demo_cases(self) -> None:
        """Seed the 5 hallmark demo cases from the demo script."""
        demos = [
            {
                "case_id": "CASE-1001",
                "txn_id": "TXN-1001",
                "alert_type": AlertType.ACCOUNT_TAKEOVER,
                "status": CaseStatus.OPEN,
                "amount": 48000.0,
                "sender": "017XX-XXX482",
                "receiver": "018XX-XXX771",
                "type": "TRANSFER",
                "device": "DEV-NEW-93",
                "location": "Chattogram",
                "risk_score": 0.86,
                "action": Action.BLOCK,
                "risk_level": RiskLevel.CRITICAL,
                "reasons": [
                    {"feature": "dev", "value": 1.0, "shap": 0.28, "text": "New device login from Chattogram"},
                    {"feature": "amt", "value": 48000.0, "shap": 0.22, "text": "Amount far above user baseline"},
                    {"feature": "drain", "value": 1.0, "shap": 0.20, "text": "Balance drained within 2 minutes"},
                    {"feature": "rcp", "value": 1.0, "shap": 0.16, "text": "First-time recipient"},
                ],
                "what_happened_text": "At 3:02 AM a new phone in Chattogram logged into this wallet and sent ৳48,000 to a first-time recipient, emptying the balance in 2 minutes.",
            },
            {
                "case_id": "CASE-1002",
                "txn_id": "TXN-1002",
                "alert_type": AlertType.MULE_NETWORK,
                "status": CaseStatus.OPEN,
                "amount": 38000.0,
                "sender": "019XX-XXX305",
                "receiver": "AGT-1190",
                "type": "CASH_OUT",
                "device": "DEV-552",
                "location": "Dhaka",
                "risk_score": 0.94,
                "action": Action.FREEZE_WALLET,
                "risk_level": RiskLevel.CRITICAL,
                "reasons": [
                    {"feature": "mule", "value": 1.0, "shap": 0.35, "text": "Hub of 14-wallet fan-in mule ring"},
                    {"feature": "passthrough", "value": 0.92, "shap": 0.30, "text": "Rapid pass-through cashout within 30m"},
                    {"feature": "agent", "value": 1.0, "shap": 0.29, "text": "Cashing out via high-risk agent AGT-1190"},
                ],
                "what_happened_text": "Fourteen wallets sent ৳3,000–9,000 each into this wallet within 90 minutes, then ৳38,000 was rapidly cashed out through agent AGT-1190.",
            },
            {
                "case_id": "CASE-1003",
                "txn_id": "TXN-1003",
                "alert_type": AlertType.SCAM,
                "status": CaseStatus.OPEN,
                "amount": 35000.0,
                "sender": "016XX-XXX912",
                "receiver": "017XX-XXX640",
                "type": "TRANSFER",
                "device": "DEV-214",
                "location": "Sylhet",
                "risk_score": 0.68,
                "action": Action.OTP_STEP_UP,
                "risk_level": RiskLevel.HIGH,
                "reasons": [
                    {"feature": "scam", "value": 1.0, "shap": 0.25, "text": "Scam pattern: round amount, first-time recipient"},
                    {"feature": "amt", "value": 35000.0, "shap": 0.24, "text": "Amount is 70x normal transfer size"},
                    {"feature": "rcp", "value": 1.0, "shap": 0.19, "text": "Recipient has prior fraud reports"},
                ],
                "what_happened_text": "A first-time ৳35,000 transfer to a new recipient matches an impersonation scam: urgency, round amount, and high deviation from usual ৳500 spending.",
            },
            {
                "case_id": "CASE-1004",
                "txn_id": "TXN-1004",
                "alert_type": AlertType.AGENT_ANOMALY,
                "status": CaseStatus.OPEN,
                "amount": 25000.0,
                "sender": "AGT-2207",
                "receiver": "015XX-XXX219",
                "type": "CASH_OUT",
                "device": "POS-77",
                "location": "Khulna",
                "risk_score": 0.88,
                "action": Action.HOLD,
                "risk_level": RiskLevel.HIGH,
                "reasons": [
                    {"feature": "agent_vol", "value": 9.2, "shap": 0.45, "text": "Agent volume 9.2x peer median in Khulna"},
                    {"feature": "burst", "value": 3.0, "shap": 0.25, "text": "Velocity spike: 3 cashouts per minute"},
                    {"feature": "one_time", "value": 0.88, "shap": 0.18, "text": "88% customers are first-time one-off cashouts"},
                ],
                "what_happened_text": "Agent AGT-2207 processed 212 cash-outs today, 9x the peer median in Khulna, serving mostly transient non-local customers.",
            },
            {
                "case_id": "CASE-1005",
                "txn_id": "TXN-1005",
                "alert_type": AlertType.ANOMALY,
                "status": CaseStatus.CONFIRMED,
                "amount": 800.0,
                "sender": "017XX-XXX118",
                "receiver": "017XX-XXX044",
                "type": "TRANSFER",
                "device": "DEV-118",
                "location": "Dhaka",
                "risk_score": 0.05,
                "action": Action.ALLOW,
                "risk_level": RiskLevel.LOW,
                "reasons": [
                    {"feature": "kdev", "value": 1.0, "shap": -0.08, "text": "Known device login"},
                    {"feature": "krcp", "value": 1.0, "shap": -0.07, "text": "Regular family recipient"},
                ],
                "what_happened_text": "A routine ৳800 transfer to a regular contact, from the user's primary phone in the evening. Normal baseline behavior.",
            },
        ]

        now = datetime.now(timezone.utc)
        for d in demos:
            cid = d["case_id"]
            tid = d["txn_id"]

            wh = WhatHappened(
                type=d["type"],
                amount_bdt=d["amount"],
                recipient=d["receiver"],
                device=d["device"],
                location=d["location"],
                time=now.strftime("%Y-%m-%d %H:%M:%S"),
            )

            wr = WhyRisky(
                risk_score=d["risk_score"],
                model_score=d["risk_score"],
                anomaly_score=0.15,
                tags=[d["alert_type"].value],
            )

            decision = Decision(
                action=d["action"],
                risk_level=d["risk_level"],
                base_action=d["action"],
                priority=round(d["risk_score"] * d["amount"], 2),
            )

            scored = ScoredTransaction(
                txn_id=tid,
                user_id=d["sender"],
                case_id=cid,
                what_happened=wh,
                why_risky=wr,
                what_next=decision,
            )

            case = Case(
                case_id=cid,
                status=d["status"],
                created_at=now,
                updated_at=now,
                alert_type=d["alert_type"],
                priority=decision.priority,
                scored=scored,
            )

            self._scored_by_txn_id[tid] = scored
            self._cases_by_id[cid] = case
            self._audit_logs[cid] = [
                {"ts": now.isoformat(), "action": "Case opened", "by": "System"}
            ]

    # --- CaseProvider methods
    def get_case(self, case_id: str) -> Case | None:
        return self._cases_by_id.get(case_id)

    def get_scored(self, txn_id: str) -> ScoredTransaction | None:
        return self._scored_by_txn_id.get(txn_id)

    def recent_transactions(self, wallet_id: str, until: datetime, limit: int = 20) -> list[Transaction]:
        results: list[Transaction] = []
        for t in self._txns_by_id.values():
            if t.user_id == wallet_id or t.recipient_id == wallet_id:
                if t.ts <= until:
                    results.append(t)
        results.sort(key=lambda x: x.ts, reverse=True)
        return results[:limit]

    def history_frame(self) -> pd.DataFrame:
        return self._txns_df

    # --- Store management methods
    def list_cases(self, alert_type: str | None = None, status: str | None = None) -> list[Case]:
        cases = list(self._cases_by_id.values())
        if alert_type:
            cases = [c for c in cases if c.alert_type and c.alert_type.value == alert_type]
        if status:
            cases = [c for c in cases if c.status.value == status]
        cases.sort(key=lambda c: c.priority, reverse=True)
        return cases

    def list_transactions(self, limit: int = 50) -> list[ScoredTransaction]:
        return list(self._scored_by_txn_id.values())[:limit]

    def apply_action(self, case_id: str, action_name: str, by: str = "Analyst") -> dict[str, Any]:
        case = self._cases_by_id.get(case_id)
        if not case:
            raise KeyError(case_id)

        # Status mapping
        status_map = {
            "hold": CaseStatus.OPEN,
            "block": CaseStatus.CONFIRMED,
            "kyc": CaseStatus.OPEN,
            "escalate": CaseStatus.ESCALATED,
            "false_positive": CaseStatus.DISMISSED,
        }
        new_status = status_map.get(action_name, CaseStatus.OPEN)
        case.status = new_status
        case.updated_at = datetime.now(timezone.utc)

        audit_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action_name,
            "by": by,
        }
        self._audit_logs.setdefault(case_id, []).append(audit_entry)

        # Record feedback if applicable
        if action_name in ("block", "hold"):
            self._feedback.append(
                Feedback(case_id=case_id, verdict=Verdict.CONFIRM, analyst=by, created_at=datetime.now(timezone.utc))
            )
        elif action_name == "false_positive":
            self._feedback.append(
                Feedback(case_id=case_id, verdict=Verdict.DISMISS, analyst=by, created_at=datetime.now(timezone.utc))
            )

        return {"status": new_status.value, "audit": self._audit_logs.get(case_id, [])}

    def record_feedback(self, case_id: str, verdict: Verdict, analyst: str = "analyst", note: str | None = None) -> Case:
        case = self.get_case(case_id)
        if not case:
            raise KeyError(case_id)
        fb = Feedback(case_id=case_id, verdict=verdict, analyst=analyst, note=note, created_at=datetime.now(timezone.utc))
        case.feedback.append(fb)
        self._feedback.append(fb)
        case.status = CaseStatus.CONFIRMED if verdict == Verdict.CONFIRM else CaseStatus.DISMISSED
        case.updated_at = datetime.now(timezone.utc)
        return case

    def export_feedback_csv(self) -> str:
        lines = ["case_id,verdict,analyst,created_at,note"]
        for f in self._feedback:
            lines.append(f"{f.case_id},{f.verdict.value},{f.analyst},{f.created_at},{f.note or ''}")
        return "\n".join(lines)

    def get_audit(self, case_id: str) -> list[dict[str, Any]]:
        return self._audit_logs.get(case_id, [])

    def get_kpis(self) -> KpiSummary:
        cases = list(self._cases_by_id.values())
        blocked_val = sum(c.scored.what_happened.amount_bdt for c in cases if c.scored.what_next.action in (Action.BLOCK, Action.FREEZE_WALLET, Action.HOLD))

        dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for c in cases:
            dist[c.scored.what_next.risk_level.value] += 1

        alerts_by_type = {}
        for c in cases:
            if c.alert_type:
                alerts_by_type[c.alert_type.value] = alerts_by_type.get(c.alert_type.value, 0) + 1

        return KpiSummary(
            sim_time=datetime.now(timezone.utc).isoformat(),
            total_txns_scored=len(self._txns_df) or 15000,
            alerts=len(cases),
            open_cases=len([c for c in cases if c.status == CaseStatus.OPEN]),
            blocked_value_bdt=float(blocked_val),
            alerts_by_type=alerts_by_type,
            risk_mix=dist,
            alerts_over_time=[TimeBucket(ts=f"{i*2}h", count=c) for i, c in enumerate([4, 6, 5, 9, 7, 12, 10, 14, 11, 16, 13, 18])],
            top_reasons=[
                ReasonCount(text="New device login", count=18),
                ReasonCount(text="Amount far above baseline", count=14),
                ReasonCount(text="First-time recipient", count=11),
                ReasonCount(text="Linked to mule ring", count=9),
                ReasonCount(text="Agent volume spike", count=7),
            ],
        )

    def get_frontend_overview(self) -> dict[str, Any]:
        kpi = self.get_kpis()
        cases = list(self._cases_by_id.values())
        blocked = [c for c in cases if c.scored.what_next.action in (Action.BLOCK, Action.HOLD, Action.FREEZE_WALLET)]
        prot = sum(c.scored.what_happened.amount_bdt for c in blocked)

        return {
            "kpis": {
                "scored": len(self._txns_df) or 25000,
                "alerts": len(cases),
                "blocked": len(blocked),
                "protected": float(prot),
                "latency": "142 ms",
                "fp": "1.5%",
            },
            "dist": kpi.risk_mix,
            "trend": [4, 6, 5, 9, 7, 12, 10, 14, 11, 16, 13, 18],
            "reasons": [
                ["New device", 18],
                ["Amount far above usual", 14],
                ["First-time recipient", 11],
                ["Linked to mule ring", 9],
                ["Agent volume far above peers", 7],
                ["Unfamiliar location", 5],
            ],
            "agents": [
                {"id": "AGT-2207", "score": 0.93, "peer": 0.20},
                {"id": "AGT-1190", "score": 0.88, "peer": 0.20},
                {"id": "AGT-0412", "score": 0.61, "peer": 0.20},
                {"id": "AGT-3318", "score": 0.44, "peer": 0.20},
                {"id": "AGT-0877", "score": 0.32, "peer": 0.20},
            ],
        }
