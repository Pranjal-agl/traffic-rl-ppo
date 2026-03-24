"""
train.py — Train a PPO agent on the TrafficEnv.

Usage:
    python train.py

Outputs:
    models/ppo_traffic.zip   — saved policy
    logs/                    — tensorboard logs
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from env.traffic_env import TrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000
N_ENVS          = 4          # parallel envs for faster training
MODEL_PATH      = "models/ppo_traffic"
LOG_DIR         = "logs/"
EVAL_FREQ       = 10_000

os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


def make_env():
    return Monitor(TrafficEnv())


def train():
    print("=" * 55)
    print("  Traffic Signal RL — PPO Training")
    print("=" * 55)

    # vectorised env for parallel rollouts
    vec_env  = make_vec_env(make_env, n_envs=N_ENVS)
    eval_env = Monitor(TrafficEnv())

    # stop early if mean reward exceeds threshold
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=-50, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        callback_on_new_best=stop_cb,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,        # encourages exploration
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs …\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)

    model.save(MODEL_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}.zip")

    # ── quick eval ────────────────────────────────────────────────────────────
    print("\nRunning 5-episode evaluation …")
    rewards = []
    for _ in range(5):
        obs, _ = eval_env.reset()
        total  = 0.0
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = eval_env.step(int(action))
            total += r
            done   = terminated or truncated
        rewards.append(total)

    print(f"Mean episode reward : {np.mean(rewards):.1f}")
    print(f"Std                 : {np.std(rewards):.1f}")
    print("\nDone! Run  python visualize.py  to watch the agent.")


if __name__ == "__main__":
    train()
