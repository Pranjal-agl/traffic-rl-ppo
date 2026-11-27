"""
evaluate.py — Compare the trained PPO agent against tuned baseline controllers.

Every controller is run on exactly the same episodes (same seeds → same car
arrivals), so differences are due to the controller, not luck. Results are
reported with 95% confidence intervals and a paired PPO-vs-best-baseline test.

Usage:
    python evaluate.py                       # models/best_model.zip, 50 episodes/scenario
    python evaluate.py --episodes 100 --model models/ppo_traffic.zip
    python evaluate.py --retune              # re-run the baseline grid search first

Outputs:
    results/eval_results.json   raw per-episode metrics
    results/results.md          summary table (paste-ready for the README)
    metrics.png                 delay-per-vehicle comparison chart
    training_curve.png          validation delay over training (if available)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from baselines import SB3Policy, load_baselines, run_episode
from env.traffic_env import SCENARIOS

SCENARIO_ORDER = ["random", *SCENARIOS]
SCENARIO_LABELS = {
    "random": "Mixed\n(random demand)",
    "balanced": "Balanced",
    "asymmetric": "Asymmetric",
    "rush_hour": "Rush hour",
    "heavy": "Heavy",
}
METRICS = ["avg_delay", "avg_queue", "throughput", "rejected", "switches"]

# categorical slots 1–3 of the validated reference palette (fixed order)
COLORS = {"Fixed-time": "#257de0", "Max-pressure": "#f47230", "PPO": "#00b894"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def evaluate(policies, seeds):
    results = {}
    for sc in SCENARIO_ORDER:
        results[sc] = {}
        for p in policies:
            eps = [run_episode(p, s, demand=sc) for s in seeds]
            results[sc][p.name] = {m: [e[m] for e in eps] for m in METRICS}
    return results


def summarise(results):
    lines = ["| Scenario | Controller | Delay / vehicle | Avg queue | Spilled-back cars | Phase switches |",
             "|---|---|---|---|---|---|"]
    print()
    for sc, by_pol in results.items():
        print(f"── {sc} " + "─" * (60 - len(sc)))
        for name, m in by_pol.items():
            d, q, r, s = (np.mean(m[k]) for k in ["avg_delay", "avg_queue", "rejected", "switches"])
            print(f"  {name:13s} delay {d:6.2f} ± {ci95(m['avg_delay']):4.2f}   queue {q:6.2f}   "
                  f"spilled {r:6.1f}   switches {s:5.1f}")
            lines.append(f"| {SCENARIO_LABELS[sc].replace(chr(10), ' ')} | {name} | "
                         f"{d:.2f} ± {ci95(m['avg_delay']):.2f} | {q:.2f} | {r:.1f} | {s:.0f} |")

        if "PPO" in by_pol:
            base = {k: v for k, v in by_pol.items() if k != "PPO"}
            best = min(base, key=lambda k: np.mean(base[k]["avg_delay"]))
            diff = np.array(by_pol["PPO"]["avg_delay"]) - np.array(base[best]["avg_delay"])
            rel = -diff.mean() / np.mean(base[best]["avg_delay"]) * 100
            sig = "significant" if abs(diff.mean()) > ci95(diff) else "not significant"
            word = "lower" if rel > 0 else "higher"
            print(f"  → PPO delay {abs(rel):.1f}% {word} than best baseline ({best}); "
                  f"paired diff {diff.mean():+.2f} ± {ci95(diff):.2f} ({sig})")
    return "\n".join(lines)


def plot_delay(results, path="metrics.png"):
    names = list(next(iter(results.values())).keys())
    x = np.arange(len(results))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    for i, name in enumerate(names):
        means = [np.mean(results[sc][name]["avg_delay"]) for sc in results]
        errs = [ci95(results[sc][name]["avg_delay"]) for sc in results]
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, means, w * 0.92, color=COLORS.get(name, "#888888"), label=name, zorder=2)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=TEXT2, elinewidth=1, capsize=2, zorder=3)
        for xi, m, e in zip(xs, means, errs):
            ax.text(xi, m + e + 0.6, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_xticks(x, [SCENARIO_LABELS[sc] for sc in results], color=TEXT)
    ax.set_ylabel("Average delay per vehicle (steps)  — lower is better", color=TEXT2)
    ax.set_title("PPO vs tuned baselines, same traffic for every controller (95% CI)",
                 loc="left", color=TEXT, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, ncol=len(names), loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Plot saved → {path}")


def plot_training_curve(csv_path, baselines_summary, path="training_curve.png"):
    if not os.path.exists(csv_path):
        return
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(data[:, 0] / 1e6, data[:, 1], color=COLORS["PPO"], lw=2, label="PPO (validation)")
    for name, val in baselines_summary.items():
        ax.axhline(val, color=COLORS.get(name, "#888"), lw=1.5, ls="--", label=name)
    ax.set_xlabel("Training steps (millions)", color=TEXT2)
    ax.set_ylabel("Delay per vehicle (steps)", color=TEXT2)
    ax.set_title("Learning curve on random demand", loc="left", color=TEXT, fontweight="bold")
    ax.grid(color=GRID)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    lo = min(data[:, 1].min(), *baselines_summary.values())
    ax.set_ylim(0, max(lo * 3, max(baselines_summary.values()) * 1.6))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Plot saved → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model.zip")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--retune", action="store_true")
    args = ap.parse_args()

    policies = load_baselines(retune=args.retune)
    if os.path.exists(args.model):
        policies.append(SB3Policy(PPO.load(args.model), name="PPO"))
    else:
        print(f"No model at {args.model} — evaluating baselines only. Run  python train.py  first.")

    seeds = list(range(args.episodes))  # test seeds; tuning/validation use disjoint ranges
    print(f"Evaluating {len(policies)} controllers × {len(SCENARIO_ORDER)} scenarios × {len(seeds)} episodes …")
    results = evaluate(policies, seeds)
    table = summarise(results)

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w") as f:
        json.dump(results, f)
    with open("results/results.md", "w") as f:
        f.write(table + "\n")
    print("\n✓ Results → results/eval_results.json, results/results.md")

    plot_delay(results)
    base = {n: float(np.mean(results["random"][n]["avg_delay"])) for n in results["random"] if n != "PPO"}
    plot_training_curve(os.path.join(os.path.dirname(args.model) or ".", "val_curve.csv"), base)


if __name__ == "__main__":
    main()
