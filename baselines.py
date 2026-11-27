"""
Baseline signal controllers + a tuner, so PPO is compared against
*reasonable* engineering practice rather than a strawman.

All controllers share one interface:
    policy.reset()
    action = policy.act(obs, info)      # info comes from env.reset/step
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

from env.traffic_env import PHASE_LANES, TrafficEnv

TUNE_SEEDS = range(10_000, 10_040)   # disjoint from evaluation seeds
PARAMS_PATH = os.path.join("results", "baseline_params.json")


class Policy:
    name = "policy"

    def reset(self):
        pass

    def act(self, obs, info) -> int:
        raise NotImplementedError


@dataclass
class FixedTimePolicy(Policy):
    """Classic pre-timed plan: hold each phase for a fixed green time."""
    green_ns: int = 20
    green_ew: int = 20
    name: str = "Fixed-time"

    def act(self, obs, info):
        phase = info["phase"]
        green = self.green_ns if phase == 0 else self.green_ew
        return 1 - phase if info["time_in_phase"] >= green else phase


@dataclass
class MaxPressurePolicy(Policy):
    """
    Queue-actuated / max-pressure controller: switch to the other phase when
    its queue exceeds the current phase's queue by `threshold` vehicles,
    or when the current green has run `max_green` steps and someone is waiting.
    (The env enforces min-green and yellow on every controller.)
    """
    threshold: float = 2.0
    max_green: int = 60
    name: str = "Max-pressure"

    def act(self, obs, info):
        phase = info["phase"]
        q = info["queues"]
        cur = q[PHASE_LANES[phase]].sum()
        other = q[PHASE_LANES[1 - phase]].sum()
        if other > cur + self.threshold:
            return 1 - phase
        if info["time_in_phase"] >= self.max_green and other > 0:
            return 1 - phase
        return phase


class SB3Policy(Policy):
    def __init__(self, model, name="PPO"):
        self.model, self.name = model, name

    def act(self, obs, info):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_episode(policy: Policy, seed: int, demand="random", env_kwargs=None, frame_cb=None) -> dict:
    env = TrafficEnv(**(env_kwargs or {}), render_mode="rgb_array" if frame_cb else None)
    obs, info = env.reset(seed=seed, options={"demand": demand})
    policy.reset()
    done = False
    total = 0.0
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs, info))
        total += r
        done = term or trunc
        if frame_cb:
            frame_cb(env)
    return {"reward": total, **env.episode_metrics()}


def mean_delay(policy, seeds, demand="random") -> float:
    return float(np.mean([run_episode(policy, s, demand)["avg_delay"] for s in seeds]))


def tune_baselines(seeds=TUNE_SEEDS, verbose=True) -> dict:
    """Grid-search baseline parameters on random demand (tuning seeds only)."""
    best = {}

    greens = [8, 12, 16, 20, 25, 30, 40]
    scores = {(g1, g2): mean_delay(FixedTimePolicy(g1, g2), seeds) for g1, g2 in itertools.product(greens, greens)}
    (g1, g2), d = min(scores.items(), key=lambda kv: kv[1])
    best["fixed_time"] = {"green_ns": g1, "green_ew": g2, "tune_delay": d}

    grid = list(itertools.product([0, 1, 2, 3, 4, 6, 8], [20, 30, 45, 60, 90]))
    scores = {(t, m): mean_delay(MaxPressurePolicy(t, m), seeds) for t, m in grid}
    (t, m), d = min(scores.items(), key=lambda kv: kv[1])
    best["max_pressure"] = {"threshold": t, "max_green": m, "tune_delay": d}

    if verbose:
        print("Tuned baselines:", json.dumps(best, indent=2))
    return best


def load_baselines(retune=False) -> list[Policy]:
    if retune or not os.path.exists(PARAMS_PATH):
        params = tune_baselines()
        os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
        with open(PARAMS_PATH, "w") as f:
            json.dump(params, f, indent=2)
    else:
        with open(PARAMS_PATH) as f:
            params = json.load(f)
    ft, mp = params["fixed_time"], params["max_pressure"]
    return [
        FixedTimePolicy(ft["green_ns"], ft["green_ew"]),
        MaxPressurePolicy(mp["threshold"], mp["max_green"]),
    ]


if __name__ == "__main__":
    load_baselines(retune=True)
