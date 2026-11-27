"""
train.py — Train a PPO agent on TrafficEnv.

Usage:
    python train.py                          # defaults below
    python train.py --timesteps 2000000 --seed 1 --n-envs 8

Outputs (under --out, default models/):
    best_model.zip        — best checkpoint by validation delay/vehicle
    ppo_traffic.zip       — final policy
    vecnormalize.pkl      — reward-normalisation stats (needed only to resume training)
    val_curve.csv         — validation delay over training
    logs/                 — TensorBoard logs (if tensorboard is installed)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from baselines import SB3Policy, run_episode
from env.traffic_env import TrafficEnv

VAL_SEEDS = range(20_000, 20_030)   # fixed validation set (disjoint from tuning & test seeds)


class DelayEvalCallback(BaseCallback):
    """
    Every `eval_freq` env steps, run the deterministic policy on a *fixed* set of
    validation seeds and keep the checkpoint with the lowest mean delay per
    vehicle. A fixed set makes checkpoints comparable (SB3's EvalCallback draws
    new random episodes each time, so "best" is partly luck).
    """

    def __init__(self, eval_freq: int, save_path: str, seeds=VAL_SEEDS):
        super().__init__()
        self.eval_freq, self.save_path, self.seeds = eval_freq, save_path, list(seeds)
        self.best = np.inf
        self.last_eval = 0
        self.history = []

    def _evaluate(self):
        policy = SB3Policy(self.model)
        delays = [run_episode(policy, s)["avg_delay"] for s in self.seeds]
        d = float(np.mean(delays))
        self.history.append((self.num_timesteps, d))
        self.logger.record("eval/avg_delay", d)
        tag = ""
        if d < self.best:
            self.best = d
            self.model.save(os.path.join(self.save_path, "best_model"))
            tag = "  ← new best, saved"
        print(f"[eval] {self.num_timesteps:>9,} steps  val delay/vehicle {d:6.2f}{tag}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._evaluate()
        return True

    def _on_training_end(self):
        self._evaluate()
        np.savetxt(os.path.join(self.save_path, "val_curve.csv"), np.array(self.history),
                   delimiter=",", header="timesteps,val_avg_delay", comments="", fmt=["%d", "%.4f"])


def linear_schedule(start: float, end: float = 0.0):
    # SB3 passes progress_remaining: 1 → 0 over training
    return lambda p: end + p * (start - end)


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        print("(tensorboard not installed — skipping TensorBoard logging)")
        return False


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models")
    p.add_argument("--eval-freq", type=int, default=50_000, help="in total env steps")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (small nets are faster with 1)")
    return p.parse_args()


def train(args):
    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, "logs")

    print("=" * 60)
    print("  Traffic Signal RL — PPO Training")
    print("=" * 60)

    # Training envs see a freshly sampled demand scenario every episode
    # (domain randomisation), so the policy must adapt rather than memorise.
    venv = make_vec_env(TrafficEnv, n_envs=args.n_envs, seed=args.seed)
    # Observations are already scaled to [0, 1] by the env; only normalise
    # rewards (episode returns are large negative numbers otherwise).
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)

    eval_cb = DelayEvalCallback(args.eval_freq, args.out)

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=linear_schedule(3e-4, 3e-5),
        n_steps=1024,               # × n_envs = 8192 samples per update
        batch_size=512,
        n_epochs=10,
        gamma=0.995,                # delays matter over ~100s of steps
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[128, 128])),
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir if _has_tensorboard() else None,
    )

    print(f"\nTraining for {args.timesteps:,} timesteps across {args.n_envs} envs …\n")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)

    final_path = os.path.join(args.out, "ppo_traffic")
    model.save(final_path)
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"\n✓ Final model → {final_path}.zip")
    print(f"✓ Best model  → {os.path.join(args.out, 'best_model.zip')}")

    # quick check of the best checkpoint on unseen seeds
    best = PPO.load(os.path.join(args.out, "best_model"))
    delays = [run_episode(SB3Policy(best), seed=s)["avg_delay"] for s in range(20)]
    print(f"\nBest model, 20 random-demand episodes: avg delay/vehicle {np.mean(delays):.2f} ± {np.std(delays):.2f} steps")
    print("Run  python evaluate.py  for the full comparison against baselines.")


if __name__ == "__main__":
    train(parse_args())
