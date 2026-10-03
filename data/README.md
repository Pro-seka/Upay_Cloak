# Sample dataset

Our own synthetic, Bangladesh-flavoured mobile-money dataset (BDT amounts, Dhaka/Chattogram-style
locations, salary-day spikes, bKash/Nagad-like transaction types). It is committed to git as the
organizers asked, and is fully reproducible with `python scripts/generate_data.py --seed 42`.

| File | Rows | Period | Purpose |
|---|---|---|---|
| `train.csv` | ~22.5k | first 70% of the timeline (Jan 1-23) | model training + validation |
| `test.csv` | ~9.6k | last 30% (Jan 23-30) | final, untouched evaluation |

The split is **time-based** (no shuffling), so the model never sees the future.

## Columns
| Column | Meaning |
|---|---|
| `txn_id` | unique transaction id |
| `ts` | timestamp |
| `user_id` | sender wallet |
| `type` | CASH_IN, CASH_OUT, TRANSFER, PAYMENT |
| `amount` | BDT |
| `recipient_id` | receiving wallet / agent / merchant |
| `agent_id` | agent for cash-in/out (empty otherwise) |
| `device_id`, `location` | device used and city |
| `balance_before` | sender balance before the transaction |
| `is_fraud` | label (about 6% of true fraud is deliberately left unlabeled to mimic unreported fraud) |
| `scenario` | ground-truth scenario: normal, ato, scam_victim, mule_passthrough, structuring, rogue_agent (analysis only, never a model input) |

Fraud rate is about 3% (higher than real life, to give the models enough positives).
