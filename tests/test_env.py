import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium.utils.env_checker import check_env

from baselines import FixedTimePolicy, MaxPressurePolicy, run_episode
from env.traffic_env import SCENARIOS, TrafficEnv


def test_gymnasium_api():
    check_env(TrafficEnv(), skip_render_check=True)


def test_time_limit_is_truncation_not_termination():
    env = TrafficEnv(max_steps=10)
    env.reset(seed=0)
    for i in range(10):
        _, _, term, trunc, _ = env.step(0)
        assert not term
        assert trunc == (i == 9)


def test_seeding_is_deterministic():
    a = run_episode(MaxPressurePolicy(), seed=123, demand="random")
    b = run_episode(MaxPressurePolicy(), seed=123, demand="random")
    assert a == b


def test_vehicle_conservation():
    env = TrafficEnv()
    env.reset(seed=1, options={"demand": "heavy"})
    rng = np.random.default_rng(0)
    for _ in range(500):
        env.step(int(rng.integers(2)))
        s = env.stats
        assert s["arrived"] == s["accepted"] + s["rejected"]
        assert s["accepted"] == s["departed"] + env.queues.sum()
    assert (env.queues >= 0).all() and (env.queues <= env.cfg.queue_cap).all()


def test_observation_stays_in_bounds_under_overload():
    env = TrafficEnv(demand=dict(rho=1.8, split=0.5, amp=0, period=100, phase=0, minor=(1, 1)))
    obs, _ = env.reset(seed=0)
    for _ in range(env.cfg.max_steps):
        obs, *_ = env.step(0)  # never serve E/W
        assert env.observation_space.contains(obs)


def test_switch_costs_yellow_and_respects_min_green():
    env = TrafficEnv(yellow_steps=3, min_green=5)
    env.reset(seed=0)
    start = env.phase
    # switching is allowed right after reset; it costs 3 all-red steps
    for _ in range(3):
        env.step(1 - start)
        assert not env.last_green.any()
    assert env.phase == 1 - start and not env.in_yellow
    # requests to switch back are ignored until 5 green steps have been served
    switches = env.stats["switches"]
    for _ in range(5):
        env.step(start)
        assert env.last_green.any()
    assert env.stats["switches"] == switches
    env.step(start)
    assert env.stats["switches"] == switches + 1 and env.in_yellow


def test_no_discharge_during_yellow():
    env = TrafficEnv(yellow_steps=3, min_green=0)
    env.reset(seed=0)
    env.queues[:] = 10
    start = env.phase
    departed = env.stats["departed"]
    for _ in range(3):
        env.step(1 - start)
    assert env.stats["departed"] == departed


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_baselines_run_on_every_scenario(scenario):
    m = run_episode(FixedTimePolicy(15, 15), seed=0, demand=scenario)
    assert m["throughput"] > 0
