"""
Single 4-way intersection with realistic signal switching.

Lanes:   0 = North, 1 = South, 2 = East, 3 = West
Phases:  0 = North/South green, 1 = East/West green

What makes this non-trivial (and different from the v1 env):
  * Changing phase costs a yellow / all-red clearance interval during which
    nobody moves, and each green must last at least ``min_green`` steps.
    Without this, the optimal policy is simply "flip the light every step".
  * Demand differs per lane and drifts over the episode (e.g. a rush hour
    that shifts from North/South to East/West), so a fixed timing plan
    cannot be right all the time.
  * Queues are finite; arrivals that find a full queue are rejected
    (spill-back) and penalised, so the agent can't hide congestion by
    letting a lane overflow.

One step ~ one second-ish of simulated time; each green lane discharges
``discharge`` vehicle(s) per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

NUM_LANES = 4
LANE_NAMES = ("N", "S", "E", "W")
PHASE_LANES = np.array(
    [
        [True, True, False, False],  # phase 0: N/S green
        [False, False, True, True],  # phase 1: E/W green
    ],
    dtype=bool,
)
NUM_PHASES = len(PHASE_LANES)

# Back-compat export used by older scripts.
MAX_QUEUE = 40


@dataclass
class EnvConfig:
    max_steps: int = 720          # episode length (truncation, not termination)
    queue_cap: int = MAX_QUEUE    # vehicles per lane before spill-back
    discharge: int = 1            # vehicles per green lane per step
    yellow_steps: int = 3         # clearance time when switching (no discharge)
    min_green: int = 5            # green must last at least this long
    overflow_penalty: float = 5.0 # extra cost per rejected (spilled-back) vehicle
    reward_scale: float = 0.1     # reward = -scale * (queued + penalty*rejected)
    red_time_norm: float = 60.0   # obs normalisation for "time since green"
    phase_time_norm: float = 60.0 # obs normalisation for "time in phase"
    ema_alpha: float = 0.05       # smoothing for the arrival-rate detector
    # Demand: "random" samples a fresh scenario each reset (used for training),
    # or a dict / scenario name from SCENARIOS for fixed evaluation conditions.
    demand: object = "random"


# Demand is described by:
#   rho    total critical flow ratio (≈ fraction of time the intersection
#          must show green; >1 − lost-time fraction means oversaturated)
#   split  share of rho going to the N/S approach (rest to E/W)
#   amp    how far the split swings over the episode (sinusoid amplitude)
#   period length of one swing, in steps
#   phase  starting offset of the swing (radians)
#   minor  ratio minor/major lane inside each approach pair
SCENARIOS = {
    "balanced":   dict(rho=0.60, split=0.50, amp=0.00, period=720, phase=0.0, minor=(1.0, 1.0)),
    "asymmetric": dict(rho=0.70, split=0.72, amp=0.00, period=720, phase=0.0, minor=(0.6, 0.6)),
    "rush_hour":  dict(rho=0.75, split=0.50, amp=0.25, period=720, phase=0.0, minor=(0.8, 0.8)),
    "heavy":      dict(rho=0.88, split=0.55, amp=0.10, period=360, phase=1.0, minor=(0.9, 0.9)),
}


def sample_random_demand(rng: np.random.Generator, max_steps: int) -> dict:
    return dict(
        rho=float(rng.uniform(0.35, 0.90)),
        split=float(rng.uniform(0.30, 0.70)),
        amp=float(rng.uniform(0.0, 0.25)),
        period=float(rng.choice([max_steps / 2, max_steps, 2 * max_steps])),
        phase=float(rng.uniform(0, 2 * np.pi)),
        minor=(float(rng.uniform(0.4, 1.0)), float(rng.uniform(0.4, 1.0))),
    )


class TrafficEnv(gym.Env):
    """
    Observation (16,), all roughly in [0, 1]:
        queues / queue_cap                       (4)
        smoothed arrival rate per lane           (4)   loop-detector estimate
        time since lane last had green / norm    (4)   clipped to 1
        current phase one-hot                    (2)
        in-yellow flag                           (1)
        time in current green / norm             (1)   clipped to 1

    Action (Discrete 2): the phase the agent wants (0 = N/S, 1 = E/W).
        Requesting the current phase holds it. Requesting the other phase
        starts a yellow interval (if min_green has elapsed); requests during
        yellow or before min_green are ignored.

    Reward: −reward_scale × (vehicles queued + overflow_penalty × rejected)
        Summed over an episode this is total vehicle delay, so maximising
        reward == minimising delay.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None, **overrides):
        super().__init__()
        cfg = config or EnvConfig()
        if overrides:
            cfg = EnvConfig(**{**asdict(cfg), **overrides})
        self.cfg = cfg
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_PHASES)
        self._renderer = None

    # ── demand ────────────────────────────────────────────────────────────────
    def _resolve_demand(self, options) -> dict:
        demand = (options or {}).get("demand", self.cfg.demand)
        if isinstance(demand, str):
            if demand == "random":
                return sample_random_demand(self.np_random, self.cfg.max_steps)
            return dict(SCENARIOS[demand])
        return dict(demand)

    def arrival_rates(self, t: Optional[int] = None) -> np.ndarray:
        d = self.demand
        t = self.step_count if t is None else t
        split = np.clip(d["split"] + d["amp"] * np.sin(2 * np.pi * t / d["period"] + d["phase"]), 0.05, 0.95)
        ns, ew = d["rho"] * split, d["rho"] * (1 - split)
        mn, me = d["minor"]
        return np.clip(np.array([ns, ns * mn, ew, ew * me]), 0.0, 0.95)

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def in_yellow(self) -> bool:
        return self.yellow_remaining > 0

    def green_mask(self) -> np.ndarray:
        if self.in_yellow:
            return np.zeros(NUM_LANES, dtype=bool)
        return PHASE_LANES[self.phase].copy()

    def _get_obs(self) -> np.ndarray:
        c = self.cfg
        obs = np.concatenate(
            [
                self.queues / c.queue_cap,
                self.arrival_ema,
                np.minimum(self.red_time / c.red_time_norm, 1.0),
                np.eye(NUM_PHASES)[self.phase],
                [float(self.in_yellow)],
                [min(self.time_in_phase / c.phase_time_norm, 1.0)],
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "phase": self.phase,
            "in_yellow": self.in_yellow,
            "time_in_phase": self.time_in_phase,
            "queues": self.queues.copy(),
            "red_time": self.red_time.copy(),
            "stats": dict(self.stats),
        }

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.demand = self._resolve_demand(options)
        self.queues = np.zeros(NUM_LANES, dtype=np.float64)
        self.red_time = np.zeros(NUM_LANES, dtype=np.float64)
        self.arrival_ema = np.zeros(NUM_LANES, dtype=np.float64)
        self.phase = int(self.np_random.integers(NUM_PHASES))
        self.next_phase = self.phase
        self.yellow_remaining = 0
        self.time_in_phase = self.cfg.min_green  # allowed to switch immediately
        self.step_count = 0
        self.last_green = self.green_mask()
        self.stats = dict(arrived=0, accepted=0, rejected=0, departed=0,
                          delay=0.0, switches=0, max_queue=0.0)
        return self._get_obs(), self._get_info()

    def step(self, action):
        c = self.cfg
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        # 1. signal controller
        if not self.in_yellow and action != self.phase and self.time_in_phase >= c.min_green:
            self.stats["switches"] += 1
            self.next_phase = action
            if c.yellow_steps > 0:
                self.yellow_remaining = c.yellow_steps
            else:
                self.phase, self.time_in_phase = action, 0

        green = self.green_mask()
        if self.in_yellow:
            self.yellow_remaining -= 1
            if self.yellow_remaining == 0:
                self.phase, self.time_in_phase = self.next_phase, 0
        else:
            self.time_in_phase += 1

        # 2. arrivals (with spill-back when the queue is full)
        rates = self.arrival_rates()
        arrivals = (self.np_random.random(NUM_LANES) < rates).astype(np.float64)
        space = c.queue_cap - self.queues
        accepted = np.minimum(arrivals, space)
        rejected = arrivals - accepted
        self.queues += accepted
        self.arrival_ema += c.ema_alpha * (arrivals - self.arrival_ema)

        # 3. departures on green lanes
        served = np.minimum(self.queues, np.where(green, c.discharge, 0))
        self.queues -= served

        # 4. bookkeeping
        self.red_time = np.where(green, 0.0, self.red_time + 1.0)
        self.step_count += 1
        self.last_green = green
        queued = float(self.queues.sum())
        n_rej = float(rejected.sum())
        s = self.stats
        s["arrived"] += int(arrivals.sum())
        s["accepted"] += int(accepted.sum())
        s["rejected"] += int(n_rej)
        s["departed"] += int(served.sum())
        s["delay"] += queued
        s["max_queue"] = max(s["max_queue"], float(self.queues.max()))

        reward = -c.reward_scale * (queued + c.overflow_penalty * n_rej)
        terminated = False                            # no terminal state
        truncated = self.step_count >= c.max_steps    # time limit → bootstrap

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from env.rendering import Renderer
            self._renderer = Renderer()
        return self._renderer.draw(self)

    # ── episode metrics ──────────────────────────────────────────────────────
    def episode_metrics(self) -> dict:
        s = self.stats
        steps = max(self.step_count, 1)
        return {
            "avg_queue": s["delay"] / steps,                         # vehicles waiting, averaged over time
            "avg_delay": s["delay"] / max(s["accepted"], 1),         # steps waited per vehicle
            "throughput": s["departed"] / steps,                     # vehicles per step
            "rejected": s["rejected"],                               # spill-back count
            "switches": s["switches"],
            "max_queue": s["max_queue"],
        }

    # kept for older callers
    def avg_queue(self) -> float:
        return self.episode_metrics()["avg_queue"]
