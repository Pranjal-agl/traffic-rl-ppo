"""
visualize.py — Watch the trained PPO agent drive the intersection.
               Saves a demo.gif automatically.

Usage:
    python visualize.py [--baseline]   # --baseline runs fixed-timer instead
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygame
from PIL import Image

from env.traffic_env import TrafficEnv, MAX_QUEUE

# ── palette ───────────────────────────────────────────────────────────────────
BG          = (15,  15,  20)
ROAD        = (45,  45,  55)
LANE_GREEN  = (50,  200, 80)
LANE_RED    = (200, 60,  60)
CAR_COL     = (255, 210, 60)
TEXT_COL    = (220, 220, 230)
ACCENT      = (100, 160, 255)

WIN_W, WIN_H = 640, 640
FPS          = 12
GIF_PATH     = "demo.gif"


# ── direction layout ──────────────────────────────────────────────────────────
# Lanes: 0=N, 1=S, 2=E, 3=W
LANE_LABELS = ["N", "S", "E", "W"]

# Bar rectangles (x, y, w, h) for each lane queue display
BAR_SPECS = {
    0: (270, 50,  100, 200),   # North  — top centre
    1: (270, 390, 100, 200),   # South  — bottom centre
    2: (390, 270, 200, 100),   # East   — right centre
    3: (50,  270, 200, 100),   # West   — left centre
}

def draw_frame(surface, env_state: dict, step: int, mode_label: str):
    surface.fill(BG)

    # ── road cross ────────────────────────────────────────────────────────────
    pygame.draw.rect(surface, ROAD, (260, 0,   120, 640))
    pygame.draw.rect(surface, ROAD, (0,   260, 640, 120))

    # ── lane bars ─────────────────────────────────────────────────────────────
    queues = env_state["queues"]
    green  = env_state["green"]

    font_sm = pygame.font.SysFont("monospace", 16, bold=True)
    font_lg = pygame.font.SysFont("monospace", 22, bold=True)

    for i, (x, y, w, h) in BAR_SPECS.items():
        q     = int(queues[i])
        ratio = q / MAX_QUEUE
        col   = LANE_GREEN if green[i] else LANE_RED

        # background track
        pygame.draw.rect(surface, (40, 40, 50), (x, y, w, h), border_radius=6)

        # filled portion
        if w > h:   # horizontal bar (E/W)
            fill_w = int(w * ratio)
            pygame.draw.rect(surface, col, (x, y, fill_w, h), border_radius=6)
        else:       # vertical bar (N/S)
            fill_h = int(h * ratio)
            pygame.draw.rect(surface, col,
                             (x, y + h - fill_h, w, fill_h), border_radius=6)

        # label + count
        lbl = font_sm.render(f"{LANE_LABELS[i]}: {q}", True, TEXT_COL)
        if w > h:
            surface.blit(lbl, (x + 4, y + h//2 - 8))
        else:
            surface.blit(lbl, (x + w//2 - 20, y + h + 4))

        # traffic light dot
        dot_x = x + w//2
        dot_y = y - 14 if h > w else y + h + 20
        pygame.draw.circle(surface, col, (dot_x, dot_y), 8)

    # ── car dots in road ─────────────────────────────────────────────────────
    for i in range(int(queues[0])):   # North
        pygame.draw.circle(surface, CAR_COL, (305 + i*8 % 20, 240 - i*8), 5)
    for i in range(int(queues[1])):   # South
        pygame.draw.circle(surface, CAR_COL, (335 - i*8 % 20, 400 + i*8), 5)
    for i in range(int(queues[2])):   # East
        pygame.draw.circle(surface, CAR_COL, (400 + i*8, 305 + i*8 % 20), 5)
    for i in range(int(queues[3])):   # West
        pygame.draw.circle(surface, CAR_COL, (240 - i*8, 335 - i*8 % 20), 5)

    # ── HUD ──────────────────────────────────────────────────────────────────
    total_q = int(queues.sum())
    hud1 = font_lg.render(f"Step {step:>4}   Queue: {total_q}", True, ACCENT)
    hud2 = font_sm.render(mode_label, True, TEXT_COL)
    surface.blit(hud1, (10, 10))
    surface.blit(hud2, (10, 38))


def run_episode(model, baseline=False):
    env    = TrafficEnv()
    obs, _ = env.reset()

    pygame.init()
    surface = pygame.Surface((WIN_W, WIN_H))
    clock   = pygame.display.set_mode((WIN_W, WIN_H), flags=pygame.NOFRAME)
    pygame.display.set_caption("Traffic RL")

    frames      = []
    total_reward = 0.0
    done        = False
    step        = 0
    phase_timer = 0
    fixed_action = 0

    label = "Agent: Fixed Timer (Baseline)" if baseline else "Agent: PPO (Trained)"

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        if baseline:
            # Fixed-timer: alternate N/S and E/W every 10 steps
            if phase_timer % 10 == 0:
                fixed_action = 1 - fixed_action
            action = fixed_action
            phase_timer += 1
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        step += 1

        green = env._green_mask(int(action))
        state = {"queues": env.queues.copy(), "green": green}

        draw_frame(surface, state, step, label)
        clock.blit(surface, (0, 0))
        pygame.display.flip()

        # capture frame for GIF
        raw = pygame.surfarray.array3d(surface)
        img = Image.fromarray(raw.transpose(1, 0, 2))
        frames.append(img.resize((320, 320), Image.LANCZOS))

        pygame.time.delay(1000 // FPS)

    pygame.quit()

    # save GIF
    path = f"{'baseline' if baseline else 'agent'}_demo.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   loop=0, duration=1000//FPS)
    print(f"✓ GIF saved → {path}")
    print(f"  Total reward: {total_reward:.1f}   Avg queue: {env.avg_queue():.2f}")
    return total_reward, env.avg_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Run fixed-timer baseline instead of trained agent")
    args = parser.parse_args()

    if not args.baseline:
        from stable_baselines3 import PPO
        if not os.path.exists("models/best_model.zip"):
            print("No model found. Run  python train.py  first.")
            sys.exit(1)
        model = PPO.load("models/best_model")
    else:
        model = None

    run_episode(model, baseline=args.baseline)
