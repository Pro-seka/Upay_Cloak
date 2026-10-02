# Data
No raw data is committed. Generate the synthetic dataset with:

    python scripts/generate_data.py

This writes `data/transactions.csv` (git-ignored). Columns: txn_id, ts, user_id, type, amount,
recipient_id, agent_id, device_id, location, balance_before, is_fraud, scenario.
