"""Action recommendation engine: risk tiers -> action, thresholds tuned by cost."""
import itertools
import numpy as np

ACTIONS = ["allow", "otp_step_up", "hold", "block"]
STOP_RATE = np.array([0.0, 0.5, 0.8, 0.95])     # share of fraud value each action prevents (assumption)
FRICTION = np.array([0.0, 20.0, 100.0, 300.0])  # BDT cost of bothering a legitimate customer (assumption)


def to_action_idx(score, th):
    return np.digitize(score, th)  # th = [t_otp, t_hold, t_block]


def total_cost(score, amount, y, th):
    a = to_action_idx(score, th)
    missed = (amount * (1 - STOP_RATE[a]))[y == 1].sum()
    friction = FRICTION[a][y == 0].sum()
    return missed + friction


def tune_thresholds(score, amount, y, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = (np.inf, None)
    for t in itertools.combinations(grid, 3):
        c = total_cost(score, amount, y, t)
        if c < best[0]:
            best = (c, [float(x) for x in t])
    return best[1]


def recommend(score, th):
    return ACTIONS[int(to_action_idx(score, th))]
