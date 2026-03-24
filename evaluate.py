"""
evaluate.py — Compare trained PPO agent vs fixed-timer baseline.
              Prints a summary table and saves a metrics plot.

Usage:
    python evaluate.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from env.traffic_env import TrafficEnv

N_EPISODES   = 20
FIXED_PERIOD = 10     # fixed-timer switches phase every N steps


def run_fixed_timer(n_episodes=N_EPISODES):
    results = []
    for _ in range(n_episodes):
        env  = TrafficEnv()
        obs, _ = env.reset()
        done = False
        step = 0
        total_r = 0.0
        action  = 0
        while not done:
            if step % FIXED_PERIOD == 0:
                action = 1 - action      # alternate 0↔1
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            done = terminated or truncated
            step += 1
        results.append({"reward": total_r, "avg_queue": env.avg_queue(),
                         "total_wait": env.total_wait()})
    return results


def run_ppo(model, n_episodes=N_EPISODES):
    results = []
    for _ in range(n_episodes):
        env  = TrafficEnv()
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(int(action))
            total_r += r
            done = terminated or truncated
        results.append({"reward": total_r, "avg_queue": env.avg_queue(),
                         "total_wait": env.total_wait()})
    return results


def summarise(label, results):
    rewards = [r["reward"]     for r in results]
    queues  = [r["avg_queue"]  for r in results]
    waits   = [r["total_wait"] for r in results]
    print(f"\n{'─'*45}")
    print(f"  {label}")
    print(f"{'─'*45}")
    print(f"  Mean episode reward : {np.mean(rewards):>8.1f}  ± {np.std(rewards):.1f}")
    print(f"  Mean avg queue      : {np.mean(queues):>8.2f}  ± {np.std(queues):.2f}")
    print(f"  Mean total wait     : {np.mean(waits):>8.1f}  ± {np.std(waits):.1f}")
    return np.mean(rewards), np.mean(queues), np.mean(waits)


def plot_comparison(fixed_res, ppo_res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("PPO Agent vs Fixed-Timer Baseline", fontsize=14, fontweight="bold")

    metrics = [
        ("reward",     "Episode Reward (higher = better)",   True),
        ("avg_queue",  "Avg Queue Length (lower = better)",  False),
        ("total_wait", "Total Wait (lower = better)",        False),
    ]

    colors = {"Fixed Timer": "#e05252", "PPO Agent": "#52a8e0"}

    for ax, (key, title, higher_better) in zip(axes, metrics):
        fixed_vals = [r[key] for r in fixed_res]
        ppo_vals   = [r[key] for r in ppo_res]

        x = np.arange(N_EPISODES)
        ax.plot(x, fixed_vals, color=colors["Fixed Timer"], alpha=0.7,
                linewidth=1.5, label="Fixed Timer")
        ax.plot(x, ppo_vals,   color=colors["PPO Agent"],   alpha=0.7,
                linewidth=1.5, label="PPO Agent")
        ax.axhline(np.mean(fixed_vals), color=colors["Fixed Timer"],
                   linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(np.mean(ppo_vals),   color=colors["PPO Agent"],
                   linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=150, bbox_inches="tight")
    print("\n✓ Plot saved → metrics.png")
    plt.show()


if __name__ == "__main__":
    model_path = "models/best_model.zip"
    if not os.path.exists(model_path):
        print("No trained model found. Run  python train.py  first.")
        sys.exit(1)

    model = PPO.load("models/best_model")

    print(f"\nEvaluating over {N_EPISODES} episodes each …")
    fixed_res = run_fixed_timer()
    ppo_res   = run_ppo(model)

    r_f, q_f, w_f = summarise("Fixed-Timer Baseline", fixed_res)
    r_p, q_p, w_p = summarise("PPO Agent",            ppo_res)

    improvement = (q_f - q_p) / q_f * 100
    print(f"\n{'='*45}")
    print(f"  Queue reduction vs baseline: {improvement:.1f}%")
    print(f"{'='*45}")

    plot_comparison(fixed_res, ppo_res)
