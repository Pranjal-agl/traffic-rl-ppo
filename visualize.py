"""
visualize.py — Record a GIF of a controller running the intersection.
Works headless (no window needed).

Usage:
    python visualize.py                              # PPO agent    → agent_demo.gif
    python visualize.py --policy max-pressure        # baseline     → max-pressure_demo.gif
    python visualize.py --policy fixed-time --scenario rush_hour --seed 3
    python visualize.py --steps 300 --every 2        # shorter GIF, keep every 2nd frame
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from baselines import SB3Policy, load_baselines, run_episode
from env.rendering import Renderer
from env.traffic_env import SCENARIOS

FPS = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["ppo", "fixed-time", "max-pressure"], default="ppo")
    ap.add_argument("--baseline", action="store_true", help="alias for --policy fixed-time (v1 compat)")
    ap.add_argument("--model", default="models/best_model.zip")
    ap.add_argument("--scenario", choices=["random", *SCENARIOS], default="rush_hour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=360, help="frames to record (episode keeps running)")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.baseline:
        args.policy = "fixed-time"

    if args.policy == "ppo":
        from stable_baselines3 import PPO
        if not os.path.exists(args.model):
            sys.exit(f"No model at {args.model}. Run  python train.py  first.")
        policy, label = SB3Policy(PPO.load(args.model)), "PPO (trained)"
        default_out = "agent_demo.gif"
    else:
        fixed, maxp = load_baselines()
        policy = fixed if args.policy == "fixed-time" else maxp
        label = f"{policy.name} (tuned)"
        default_out = f"{args.policy}_demo.gif"

    renderer = Renderer(label=f"{label} · {args.scenario}")
    frames = []

    def capture(env):
        n = env.step_count
        if n <= args.steps and n % args.every == 0:
            img = Image.fromarray(renderer.draw(env)).resize((320, 320), Image.LANCZOS)
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))

    m = run_episode(policy, args.seed, demand=args.scenario, frame_cb=capture)
    out = args.out or default_out
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=1000 // FPS * args.every, optimize=True)
    print(f"✓ GIF saved → {out}  ({len(frames)} frames)")
    print(f"  Full episode: avg delay/vehicle {m['avg_delay']:.2f}, avg queue {m['avg_queue']:.2f}, "
          f"spilled back {m['rejected']}, switches {m['switches']}")


if __name__ == "__main__":
    main()
