"""Single source of truth for evidence IDs.  Both parts MUST build ids through these helpers.

Format: <KIND>-<raw id>.   TXN ids already look like `TXN-0062110` and are used as-is.
The regex is used by the grounding validator (Part 2) and by tests in both parts.
"""
from __future__ import annotations

import re

KINDS = ("TXN", "WALLET", "DEVICE", "AGENT", "RING", "RULE", "FACTOR", "POLICY", "METRIC")

# KIND-token, where token may contain -, :, . between alphanumeric chunks (never trailing punctuation)
ID_PATTERN = re.compile(rf"\b(?:{'|'.join(KINDS)})-[A-Za-z0-9_]+(?:[-:.][A-Za-z0-9_]+)*")


def txn(txn_id: str) -> str:
    return txn_id if txn_id.startswith("TXN-") else f"TXN-{txn_id}"


def wallet(wallet_id: str) -> str:
    return f"WALLET-{wallet_id}"


def device(device_id: str) -> str:
    return f"DEVICE-{device_id}"


def agent(agent_id: str) -> str:
    return f"AGENT-{agent_id}"


def ring(ring_id: str) -> str:
    """ring_id values are already 'RING-xxxxxx'; this normalises either form."""
    return ring_id if ring_id.startswith("RING-") else f"RING-{ring_id}"


def rule(rule_id: str) -> str:
    return f"RULE-{rule_id}"


def factor(feature: str) -> str:
    return f"FACTOR-{feature}"


def policy(policy_id: str) -> str:
    return f"POLICY-{policy_id}"


def metric(name: str) -> str:
    return f"METRIC-{name}"


def extract_ids(text: str) -> list[str]:
    """All evidence-id-looking tokens in `text`, in order, de-duplicated."""
    seen: dict[str, None] = {}
    for m in ID_PATTERN.finditer(text):
        seen.setdefault(m.group(0), None)
    return list(seen)
