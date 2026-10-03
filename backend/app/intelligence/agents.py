"""Agent Risk Intelligence Service.
Profiles agent behaviour against peer baselines, detects rogue cash-out points,
structuring, night bursts, and provides risk leaderboards.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backend.app.contracts.evidence_ids import agent as eid_agent
from backend.app.contracts.interfaces import AgentRiskService
from backend.app.contracts.schemas import AgentRisk, Transaction


class PeerAgentRiskService(AgentRiskService):
    def __init__(self) -> None:
        self._agents: dict[str, AgentRisk] = {}
        self._raw_stats: dict[str, dict[str, Any]] = {}
        self._seed_demo_agents()

    def _seed_demo_agents(self) -> None:
        demo_agents = [
            ("AGT-2207", 0.93, 0.98, ["volume_spike", "night_burst", "cashout_skew"], {"volume_z": 4.2, "burst_ratio": 9.2, "night_share": 0.42, "cashout_ratio": 0.88}),
            ("AGT-1190", 0.88, 0.95, ["mule_cashout_exit", "structuring_cluster"], {"volume_z": 3.8, "burst_ratio": 6.5, "near_threshold_share": 0.35, "cashout_ratio": 0.94}),
            ("AGT-0412", 0.61, 0.75, ["cashout_skew"], {"volume_z": 1.9, "burst_ratio": 2.8, "night_share": 0.18, "cashout_ratio": 0.72}),
            ("AGT-3318", 0.44, 0.55, [], {"volume_z": 0.8, "burst_ratio": 1.4, "night_share": 0.08, "cashout_ratio": 0.55}),
            ("AGT-0877", 0.32, 0.35, [], {"volume_z": -0.2, "burst_ratio": 1.0, "night_share": 0.05, "cashout_ratio": 0.50}),
        ]
        for aid, score, pctile, flags, metrics in demo_agents:
            self._agents[aid] = AgentRisk(
                agent_id=aid,
                risk_score=score,
                peer_group="regional",
                peer_percentile=pctile,
                metrics=metrics,
                flags=flags,
                evidence_ids=[eid_agent(aid)],
            )

    def build(self, txns: pd.DataFrame) -> None:
        """Analyze agent transactions against peer group baseline."""
        if txns.empty:
            return

        df = txns.copy()
        if "agent_id" not in df.columns:
            df["agent_id"] = None

        # Filter transactions that involve agents
        agent_txns: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for _, r in df.iterrows():
            aid = r.get("agent_id")
            if pd.isna(aid) or not aid:
                # Check if recipient looks like an agent or cash_out
                rcpt = str(r.get("recipient_id", ""))
                if str(r.get("type", "")).upper() in ("CASH_OUT", "CASH_IN") or rcpt.startswith("AGT-") or rcpt.startswith("A0"):
                    aid = rcpt
                else:
                    continue

            aid_str = str(aid)
            ts = pd.to_datetime(r["ts"])
            amt = float(r["amount"])
            ttype = str(r["type"]).upper()

            agent_txns[aid_str].append({
                "ts": ts,
                "amount": amt,
                "type": ttype,
                "hour": ts.hour,
            })

        if not agent_txns:
            return

        # Calculate agent metrics
        volumes = [sum(item["amount"] for item in items) for items in agent_txns.values()]
        mean_vol = np.mean(volumes) if volumes else 1.0
        std_vol = np.std(volumes) if len(volumes) > 1 and np.std(volumes) > 0 else (mean_vol or 1.0)

        for aid, items in agent_txns.items():
            tot_amt = sum(item["amount"] for item in items)
            count = len(items)
            cashouts = sum(1 for item in items if item["type"] == "CASH_OUT")
            night_txns = sum(1 for item in items if item["hour"] < 6 or item["hour"] >= 22)
            struct_txns = sum(1 for item in items if 45000.0 <= item["amount"] < 50000.0)

            cashout_ratio = (cashouts / count) if count > 0 else 0.5
            night_share = (night_txns / count) if count > 0 else 0.0
            struct_share = (struct_txns / count) if count > 0 else 0.0
            vol_z = float((tot_amt - mean_vol) / std_vol)

            flags: list[str] = []
            if night_share >= 0.25:
                flags.append("night_burst")
            if struct_share >= 0.15:
                flags.append("structuring_cluster")
            if cashout_ratio >= 0.85 and count >= 5:
                flags.append("cashout_skew")
            if vol_z >= 2.5:
                flags.append("volume_spike")

            # Risk score calculation
            base_risk = 0.1
            if "volume_spike" in flags:
                base_risk += 0.35
            if "night_burst" in flags:
                base_risk += 0.25
            if "structuring_cluster" in flags:
                base_risk += 0.20
            if "cashout_skew" in flags:
                base_risk += 0.15

            risk_score = min(0.99, max(0.05, round(base_risk, 2)))

            self._agents[aid] = AgentRisk(
                agent_id=aid,
                risk_score=risk_score,
                peer_group="regional",
                peer_percentile=min(0.99, round(max(0.1, (risk_score * 0.9) + 0.05), 2)),
                metrics={
                    "volume_bdt": tot_amt,
                    "volume_z": round(vol_z, 2),
                    "cashout_ratio": round(cashout_ratio, 2),
                    "night_share": round(night_share, 2),
                    "near_threshold_share": round(struct_share, 2),
                },
                flags=flags,
                evidence_ids=[eid_agent(aid)],
            )

        # Re-ensure demo agents exist
        self._seed_demo_agents()

    def ingest(self, txn: Transaction) -> None:
        """Update risk when a transaction occurs at an agent."""
        aid = txn.agent_id or (txn.recipient_id if txn.type == "CASH_OUT" else None)
        if not aid:
            return
        if aid not in self._agents:
            self._agents[aid] = AgentRisk(
                agent_id=aid,
                risk_score=0.20,
                peer_group="regional",
                peer_percentile=0.25,
                metrics={"volume_bdt": txn.amount, "cashout_ratio": 1.0 if txn.type == "CASH_OUT" else 0.0},
                flags=[],
                evidence_ids=[eid_agent(aid)],
            )

    def agent_risk(self, agent_id: str, as_of: datetime | None = None) -> AgentRisk:
        if agent_id in self._agents:
            res = self._agents[agent_id]
            if as_of:
                res = res.model_copy(update={"as_of": as_of})
            return res
        return AgentRisk(
            agent_id=agent_id,
            as_of=as_of,
            risk_score=0.20,
            peer_group="regional",
            peer_percentile=0.20,
            metrics={"volume_z": 0.0, "cashout_ratio": 0.5},
            flags=[],
            evidence_ids=[eid_agent(agent_id)],
        )

    def leaderboard(self, top_n: int = 10, as_of: datetime | None = None) -> list[AgentRisk]:
        ranked = sorted(self._agents.values(), key=lambda a: a.risk_score, reverse=True)
        return ranked[:top_n]
