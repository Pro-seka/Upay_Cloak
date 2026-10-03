"""Action recommendation: risk score -> allow / otp_step_up / hold / block.

Thresholds are tuned on the validation set to minimise
    (fraud value that slips through) + (friction cost on legit customers).
The stop-rates and friction costs are ASSUMPTIONS - tune them with the team.
"""
import itertools

import numpy as np

ACTIONS = ["allow", "otp_step_up", "hold", "block"]
STOP_RATE = np.array([0.0, 0.5, 0.8, 0.95])      # share of fraud value each action prevents
FRICTION = np.array([0.0, 20.0, 100.0, 300.0])   # BDT cost of bothering a legit customer


def action_idx(score, th):
    return np.digitize(score, th)                # th = [t_otp, t_hold, t_block]


def total_cost(score, amount, y, th, w=None):
    """w = optional per-row weights (used by ml/feedback.py to up-weight analyst-reviewed cases)."""
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    a = action_idx(score, th)
    missed = ((amount * (1 - STOP_RATE[a])) * w)[y == 1].sum()
    return missed + (FRICTION[a] * w)[y == 0].sum()


def tune_thresholds(score, amount, y, w=None):
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_cost, best = np.inf, None
    for t in itertools.combinations(grid, 3):
        c = total_cost(score, amount, y, t, w)
        if c < best_cost:
            best_cost, best = c, [float(x) for x in t]
    return best


def recommend(score, th):
    return ACTIONS[int(action_idx(score, th))]
