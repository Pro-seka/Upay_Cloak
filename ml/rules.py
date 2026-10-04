"""Human-readable rule trace + scam/ATO/mule tags (feeds the case view & LLM evidence JSON).

This is the ONLY copy of rule_trace; ml.score and ml.cache import it.
"""


def rule_trace(r):
    hits = []
    def hit(rid, tag, text): hits.append(dict(rule_id=rid, tag=tag, text=text))
    if r["is_new_device"] and r["is_new_location"]:
        hit("ATO_01", "account_takeover", "New device AND new location for this wallet")
    if r["is_new_device"] and r["is_night"] and r["txn_count_1h"] >= 2:
        hit("ATO_02", "account_takeover", "New device at night with repeated transactions (velocity burst)")
    if r["type"] == "TRANSFER" and r["is_new_recipient"] and r["amount_z"] > 2.5:
        hit("SCAM_01", "scam_victim", "First-time recipient with an amount far above this user's norm")
    if r["inbound_1h"] > 0 and r["amount"] >= 0.7 * r["inbound_1h"] and r["type"] in ("TRANSFER", "CASH_OUT"):
        hit("MULE_01", "mule_passthrough", "Wallet just received money and is forwarding most of it")
    if r["device_users_count"] >= 3:
        hit("MULE_02", "mule_network", f"Device shared by {int(r['device_users_count'])} accounts")
    if r["near_thr_24h"] >= 2:
        hit("STRUCT_01", "structuring", "Repeated cash-outs just under the 50,000 BDT threshold")
    if r["agent_txn_1h"] >= 8 and (r["hour"] >= 22 or r["hour"] < 6):
        hit("AGENT_01", "agent_risk", "Agent processing an unusual late-night burst")
    return hits
