"""Evidence Builder for UpayShield.
Transforms a scored Case into a strictly structured, fact-grounded EvidenceBundle
with formal evidence IDs (TXN, WALLET, DEVICE, AGENT, RING, RULE, FACTOR, POLICY, METRIC).
"""
from __future__ import annotations

from datetime import datetime, timezone

from backend.app.contracts.evidence_ids import (
    agent as eid_agent,
    device as eid_device,
    factor as eid_factor,
    metric as eid_metric,
    policy as eid_policy,
    ring as eid_ring,
    rule as eid_rule,
    txn as eid_txn,
    wallet as eid_wallet,
)
from backend.app.contracts.interfaces import EvidenceBuilder
from backend.app.contracts.schemas import Case
from backend.app.contracts.schemas_intel import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    TimelineEvent,
)


class StructuredEvidenceBuilder(EvidenceBuilder):
    def build(self, case: Case) -> EvidenceBundle:
        scored = case.scored
        wh = scored.what_happened
        wr = scored.why_risky
        wn = scored.what_next

        items: list[EvidenceItem] = []
        what_happened_ids: list[str] = []
        why_risky_ids: list[str] = []
        what_next_ids: list[str] = []

        # --- Q1: What Happened Evidence
        tid_eid = eid_txn(scored.txn_id)
        items.append(
            EvidenceItem(
                id=tid_eid,
                kind=EvidenceKind.TXN,
                label=f"{wh.type} transaction of ৳{wh.amount_bdt:,.0f}",
                facts={
                    "txn_id": scored.txn_id,
                    "amount_bdt": float(wh.amount_bdt),
                    "type": wh.type,
                    "time": wh.time,
                    "location": wh.location,
                },
            )
        )
        what_happened_ids.append(tid_eid)

        sender_eid = eid_wallet(scored.user_id)
        items.append(
            EvidenceItem(
                id=sender_eid,
                kind=EvidenceKind.WALLET,
                label=f"Sender account {scored.user_id}",
                facts={"wallet_id": scored.user_id},
            )
        )
        what_happened_ids.append(sender_eid)

        rcpt_eid = eid_wallet(wh.recipient)
        items.append(
            EvidenceItem(
                id=rcpt_eid,
                kind=EvidenceKind.WALLET,
                label=f"Recipient account {wh.recipient}",
                facts={"recipient_id": wh.recipient},
            )
        )
        what_happened_ids.append(rcpt_eid)

        dev_eid = eid_device(wh.device)
        items.append(
            EvidenceItem(
                id=dev_eid,
                kind=EvidenceKind.DEVICE,
                label=f"Device {wh.device} in {wh.location}",
                facts={"device_id": wh.device, "location": wh.location},
            )
        )
        what_happened_ids.append(dev_eid)

        if wh.agent:
            agt_eid = eid_agent(wh.agent)
            items.append(
                EvidenceItem(
                    id=agt_eid,
                    kind=EvidenceKind.AGENT,
                    label=f"Cashout agent {wh.agent}",
                    facts={"agent_id": wh.agent},
                )
            )
            what_happened_ids.append(agt_eid)

        # --- Q2: Why Risky Evidence
        score_eid = eid_metric("risk_score")
        items.append(
            EvidenceItem(
                id=score_eid,
                kind=EvidenceKind.METRIC,
                label=f"Blended risk score {wr.risk_score:.2f} ({wn.risk_level.value})",
                facts={
                    "risk_score": round(wr.risk_score, 4),
                    "model_score": round(wr.model_score, 4),
                    "anomaly_score": round(wr.anomaly_score, 4),
                },
            )
        )
        why_risky_ids.append(score_eid)

        # Rule hits
        for rh in wr.rule_trace:
            r_eid = eid_rule(rh.rule_id)
            items.append(
                EvidenceItem(
                    id=r_eid,
                    kind=EvidenceKind.RULE,
                    label=rh.text,
                    facts={"rule_id": rh.rule_id, "tag": rh.tag},
                )
            )
            why_risky_ids.append(r_eid)

        # SHAP reason factors
        for reason in wr.reasons:
            f_eid = eid_factor(reason.feature)
            items.append(
                EvidenceItem(
                    id=f_eid,
                    kind=EvidenceKind.FACTOR,
                    label=reason.text or f"Factor {reason.feature}",
                    facts={"feature": reason.feature, "shap": round(reason.shap, 4), "value": float(reason.value)},
                )
            )
            why_risky_ids.append(f_eid)

        # Graph signals
        if wr.graph:
            if wr.graph.ring_id:
                rg_eid = eid_ring(wr.graph.ring_id)
                items.append(
                    EvidenceItem(
                        id=rg_eid,
                        kind=EvidenceKind.RING,
                        label=f"Member of mule ring {wr.graph.ring_id}",
                        facts={"ring_id": wr.graph.ring_id, "ring_score": wr.graph.ring_score, "fan_in": wr.graph.fan_in},
                    )
                )
                why_risky_ids.append(rg_eid)
            for g_flag in wr.graph.flags:
                g_eid = eid_metric(g_flag)
                if g_eid not in [i.id for i in items]:
                    items.append(
                        EvidenceItem(
                            id=g_eid,
                            kind=EvidenceKind.METRIC,
                            label=f"Graph pattern: {g_flag.replace('_', ' ')}",
                            facts={"flag": g_flag, "fan_in": wr.graph.fan_in, "passthrough_ratio": wr.graph.passthrough_ratio},
                        )
                    )
                    why_risky_ids.append(g_eid)

        # Agent risk
        if wr.agent:
            ag_eid = eid_agent(wr.agent.agent_id)
            if ag_eid not in [i.id for i in items]:
                items.append(
                    EvidenceItem(
                        id=ag_eid,
                        kind=EvidenceKind.AGENT,
                        label=f"Agent {wr.agent.agent_id} risk {wr.agent.risk_score:.2f}",
                        facts={"agent_id": wr.agent.agent_id, "risk_score": wr.agent.risk_score, "peer_percentile": wr.agent.peer_percentile},
                    )
                )
                why_risky_ids.append(ag_eid)

        # --- Q3: What Next Evidence
        pol_eid = eid_policy(f"POL_{wn.action.value.upper()}")
        items.append(
            EvidenceItem(
                id=pol_eid,
                kind=EvidenceKind.POLICY,
                label=f"Recommended action: {wn.action.value} (Priority: ৳{wn.priority:,.0f})",
                facts={
                    "recommended_action": wn.action.value,
                    "risk_level": wn.risk_level.value,
                    "base_action": wn.base_action.value,
                    "priority": float(wn.priority),
                },
            )
        )
        what_next_ids.append(pol_eid)

        for p_hit in wn.policy_trace:
            hit_eid = eid_policy(p_hit.policy_id)
            items.append(
                EvidenceItem(
                    id=hit_eid,
                    kind=EvidenceKind.POLICY,
                    label=p_hit.text,
                    facts={"policy_id": p_hit.policy_id, "raises_to": p_hit.raises_to.value if p_hit.raises_to else wn.action.value},
                )
            )
            what_next_ids.append(hit_eid)

        # Timeline Construction
        timeline = [
            TimelineEvent(
                ts=wh.time,
                txn_id=scored.txn_id,
                label=f"{wh.type} ৳{wh.amount_bdt:,.0f} to {wh.recipient}",
                amount_bdt=float(wh.amount_bdt),
                focal=True,
                evidence_ids=[tid_eid, sender_eid, rcpt_eid],
            )
        ]

        return EvidenceBundle(
            case_id=case.case_id,
            generated_at=datetime.now(timezone.utc),
            recommended_action=wn.action,
            what_happened=what_happened_ids,
            why_risky=why_risky_ids,
            what_next=what_next_ids,
            items=items,
            timeline=timeline,
        )
