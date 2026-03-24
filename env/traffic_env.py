import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ── Simulation constants ──────────────────────────────────────────────────────
MAX_QUEUE   = 10        # max cars that can queue per lane
NUM_LANES   = 4         # N, S, E, W
SPAWN_PROB  = 0.3       # probability a new car spawns each step per lane
GREEN_DUR   = 5         # how many steps a green phase lasts
MAX_STEPS   = 500       # episode length
DISCHARGE   = 2         # cars that leave per step on a green lane


class TrafficEnv(gym.Env):
    """
    4-way single intersection.

    Observation (8,):
        [queue_N, queue_S, queue_E, queue_W,
         wait_N,  wait_S,  wait_E,  wait_W]

    Action (Discrete 4):
        0 = give green to North/South pair
        1 = give green to East/West pair
        2 = give green to North only
        3 = give green to East only
        (agent picks every GREEN_DUR steps; action is held for GREEN_DUR ticks)

    Reward:
        −(total queue length) per step  →  agent minimises congestion
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()

        # observation: queues (0‥MAX_QUEUE) + cumulative waits (0‥MAX_STEPS)
        low  = np.zeros(8, dtype=np.float32)
        high = np.array([MAX_QUEUE]*4 + [MAX_STEPS]*4, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(4)

        self.reset()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _green_mask(self, action: int) -> np.ndarray:
        """Return boolean mask of which lanes are green for this action."""
        masks = {
            0: [True,  True,  False, False],   # N+S green
            1: [False, False, True,  True ],   # E+W green
            2: [True,  False, False, False],   # N only
            3: [False, False, True,  False],   # E only
        }
        return np.array(masks[action], dtype=bool)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.queues, self.waits]).astype(np.float32)

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.queues      = np.zeros(NUM_LANES, dtype=np.float32)
        self.waits       = np.zeros(NUM_LANES, dtype=np.float32)
        self.step_count  = 0
        self.current_action = 0
        self.phase_timer = 0          # counts down within a green phase
        self.history     = []         # (step, queues copy) for analysis
        return self._get_obs(), {}

    def step(self, action: int):
        self.step_count += 1

        # ── 1. spawn cars ────────────────────────────────────────────────────
        spawns = self.np_random.random(NUM_LANES) < SPAWN_PROB
        self.queues = np.minimum(self.queues + spawns.astype(float), MAX_QUEUE)

        # ── 2. discharge cars on green lanes ─────────────────────────────────
        green = self._green_mask(action)
        discharge = np.where(green, DISCHARGE, 0).astype(float)
        self.queues = np.maximum(self.queues - discharge, 0)

        # ── 3. accumulate wait for red lanes ─────────────────────────────────
        self.waits += np.where(~green, self.queues, 0)

        # ── 4. reward ────────────────────────────────────────────────────────
        reward = -float(self.queues.sum())

        # ── 5. record history ────────────────────────────────────────────────
        self.history.append({
            "step":   self.step_count,
            "queues": self.queues.copy(),
            "action": action,
            "green":  green.copy(),
        })

        terminated = self.step_count >= MAX_STEPS
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ── analysis helpers (used by evaluate.py) ───────────────────────────────
    def avg_queue(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean([h["queues"].sum() for h in self.history]))

    def total_wait(self) -> float:
        return float(self.waits.sum())
