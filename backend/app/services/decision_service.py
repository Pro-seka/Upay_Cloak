"""Policy and Decision Engine for UpayShield.
Enforces that policy rules can ONLY raise action severity, never lower it.
Records full policy trace and determines final case priority.
"""
from __future__ import annotations

from backend.app.contracts.schemas import (
    ACTION_SEVERITY,
    ALERT_TYPE_PRECEDENCE,
    SEVERITY_TO_LEVEL,
    TAG_TO_ALERT_TYPE,
    Action,
    AlertType,
    Decision,
    GraphSignals,
    PolicyHit,
    RiskLevel,
)


class DecisionEngine:
    def __init__(self, thresholds: list[float] | None = None) -> None:
        # Default thresholds [otp, hold, block]
        self.thresholds = thresholds or [0.20, 0.25, 0.30]

    def base_recommend(self, score: float) -> Action:
        t_otp, t_hold, t_block = self.thresholds
        if score >= t_block:
            return Action.BLOCK
        if score >= t_hold:
            return Action.HOLD
        if score >= t_otp:
            return Action.OTP_STEP_UP
        return Action.ALLOW

    def evaluate(
        self,
        risk_score: float,
        amount_bdt: float,
        rule_tags: list[str] = [],
        graph_signals: GraphSignals | None = None,
        agent_risk_score: float = 0.0,
        is_ring_member: bool = False,
    ) -> Decision:
        base_action = self.base_recommend(risk_score)
        current_action = base_action
        current_severity = ACTION_SEVERITY[base_action]
        policy_trace: list[PolicyHit] = []

        # Policy 1: Mule ring membership -> freeze wallet
        if is_ring_member or (graph_signals and graph_signals.ring_id):
            target = Action.FREEZE_WALLET
            if ACTION_SEVERITY[target] > current_severity:
                current_action = target
                current_severity = ACTION_SEVERITY[target]
                policy_trace.append(
                    PolicyHit(
                        policy_id="POL_RING_FREEZE",
                        text="Wallet identified as member of an active money-mule ring; auto-escalate to wallet freeze.",
                        raises_to=target,
                    )
                )

        # Policy 2: Rogue agent anomaly -> hold for human review
        if agent_risk_score >= 0.85:
            target = Action.HOLD
            if ACTION_SEVERITY[target] > current_severity:
                current_action = target
                current_severity = ACTION_SEVERITY[target]
                policy_trace.append(
                    PolicyHit(
                        policy_id="POL_AGENT_ANOMALY",
                        text="Processing agent has an anomaly score >= 0.85 vs peer median; hold transaction.",
                        raises_to=target,
                    )
                )

        # Policy 3: Account Takeover (ATO) -> block
        if "account_takeover" in rule_tags:
            target = Action.BLOCK
            if ACTION_SEVERITY[target] > current_severity:
                current_action = target
                current_severity = ACTION_SEVERITY[target]
                policy_trace.append(
                    PolicyHit(
                        policy_id="POL_ATO_BLOCK",
                        text="Account takeover signals confirmed (new device/location/night burst); block immediately.",
                        raises_to=target,
                    )
                )

        # Policy 4: Scam victim protection -> step-up verification / warn
        if "scam_victim" in rule_tags:
            target = Action.OTP_STEP_UP
            if ACTION_SEVERITY[target] > current_severity:
                current_action = target
                current_severity = ACTION_SEVERITY[target]
                policy_trace.append(
                    PolicyHit(
                        policy_id="POL_SCAM_PROTECT",
                        text="High-value transfer to unverified recipient with scam patterns; require step-up authentication.",
                        raises_to=target,
                    )
                )

        # Determine AlertType by precedence
        alert_type: AlertType | None = None
        for tag in rule_tags:
            mapped = TAG_TO_ALERT_TYPE.get(tag)
            if mapped:
                if alert_type is None or ALERT_TYPE_PRECEDENCE.index(mapped) < ALERT_TYPE_PRECEDENCE.index(alert_type):
                    alert_type = mapped

        if alert_type is None and current_action != Action.ALLOW:
            if is_ring_member or (graph_signals and "fan_in_hub" in graph_signals.flags):
                alert_type = AlertType.MULE_NETWORK
            elif agent_risk_score >= 0.85:
                alert_type = AlertType.AGENT_ANOMALY
            else:
                alert_type = AlertType.ANOMALY

        risk_level = SEVERITY_TO_LEVEL[current_severity]
        priority = round(risk_score * amount_bdt, 2)

        return Decision(
            action=current_action,
            risk_level=risk_level,
            base_action=base_action,
            alert_type=alert_type,
            policy_trace=policy_trace,
            requires_analyst=(current_action in (Action.HOLD, Action.ESCALATE)),
            priority=priority,
            thresholds_version="model-v1",
        )
