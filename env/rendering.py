"""Off-screen pygame renderer for TrafficEnv (works headless, no window needed)."""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from env.traffic_env import PHASE_LANES

BG = (15, 15, 20)
ROAD = (45, 45, 55)
TRACK = (40, 40, 50)
GREEN = (50, 200, 80)
YELLOW = (240, 190, 40)
RED = (200, 60, 60)
CAR = (255, 210, 60)
TEXT = (220, 220, 230)
ACCENT = (100, 160, 255)

SIZE = 640
LABELS = ("N", "S", "E", "W")
# queue bar rectangles (x, y, w, h)
BARS = {
    0: (270, 50, 100, 200),   # North
    1: (270, 390, 100, 200),  # South
    2: (390, 270, 200, 100),  # East
    3: (50, 270, 200, 100),   # West
}


class Renderer:
    def __init__(self, label: str = ""):
        if not pygame.font.get_init():
            pygame.font.init()
        self.surface = pygame.Surface((SIZE, SIZE))
        self.font_sm = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 26, bold=True)
        self.label = label

    def draw(self, env) -> np.ndarray:
        s = self.surface
        s.fill(BG)
        pygame.draw.rect(s, ROAD, (260, 0, 120, SIZE))
        pygame.draw.rect(s, ROAD, (0, 260, SIZE, 120))

        green = env.green_mask()
        # during clearance the outgoing phase shows yellow
        yellow_lanes = PHASE_LANES[env.phase] if env.in_yellow else np.zeros(4, bool)
        cap = env.cfg.queue_cap

        for i, (x, y, w, h) in BARS.items():
            q = int(env.queues[i])
            ratio = min(q / cap, 1.0)
            col = GREEN if green[i] else (YELLOW if yellow_lanes[i] else RED)
            pygame.draw.rect(s, TRACK, (x, y, w, h), border_radius=6)
            if w > h:  # horizontal (E/W) — fill towards the junction
                fw = int(w * ratio)
                fx = x if i == 2 else x + w - fw
                pygame.draw.rect(s, col, (fx, y, fw, h), border_radius=6)
                lbl = self.font_sm.render(f"{LABELS[i]}: {q}", True, TEXT)
                lx = x + w - lbl.get_width() - 6 if i == 2 else x + 6   # outer end, away from the signal
                s.blit(lbl, (lx, y - 22))
            else:      # vertical (N/S)
                fh = int(h * ratio)
                fy = y + h - fh if i == 0 else y
                pygame.draw.rect(s, col, (x, fy, w, fh), border_radius=6)
                lbl = self.font_sm.render(f"{LABELS[i]}: {q}", True, TEXT)
                s.blit(lbl, (x + w + 8, y + 4 if i == 0 else y + h - 20))  # outer end, beside the bar
            # signal head near the stop line
            dot = {0: (x + w // 2, 258 - 12), 1: (x + w // 2, 382 + 12), 2: (382 + 12, y + h // 2), 3: (258 - 12, y + h // 2)}[i]
            pygame.draw.circle(s, col, dot, 8)

        m = env.episode_metrics()
        state = "YELLOW" if env.in_yellow else ("N/S green" if env.phase == 0 else "E/W green")
        s.blit(self.font_lg.render(f"Step {env.step_count:>4}  Queue {int(env.queues.sum()):>3}", True, ACCENT), (10, 10))
        s.blit(self.font_sm.render(self.label, True, TEXT), (10, 42))
        s.blit(self.font_sm.render(state, True, TEXT), (10, 66))
        s.blit(self.font_sm.render(f"avg delay {m['avg_delay']:.1f}  switches {m['switches']}", True, TEXT), (10, SIZE - 26))

        return np.transpose(pygame.surfarray.array3d(s), (1, 0, 2)).copy()
