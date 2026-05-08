import gymnasium as gym
import numpy as np

from sim.vehicle_model import BicycleModel, VehicleParams
from rl.reward import compute_reward


class VehicleLateralEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        params = VehicleParams(
            m=1600.0, Iz=2500.0, lf=1.2, lr=1.6, Cf=80000.0, Cr=80000.0, vx_nominal=15.0
        )
        self.model = BicycleModel(params, dt=0.05)
        self.dt = 0.05
        self.max_steps = 400

        self.action_space = gym.spaces.Box(low=np.array([-0.4]),high=np.array([0.4]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.x = np.zeros(4, dtype=float)
        self.k = 0
        self.prev_delta = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize initial condition each episode so the policy sees the
        # full IC distribution we evaluate on (multi-IC grid + robustness
        # Cases A and B). Bounds match: ey in [-0.4, 0.4], epsi in [-0.10, 0.10].
        ey0  = float(self.np_random.uniform(-0.4, 0.4))
        eps0 = float(self.np_random.uniform(-0.10, 0.10))
        self.x = np.array([ey0, eps0, 0.0, 0.0], dtype=float)
        self.k = 0
        self.prev_delta = 0.0
        return self.x.astype(np.float32), {}

    def step(self, action):
        delta = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        self.x = self.model.step(self.x, delta)

        reward = compute_reward(self.x, delta, self.prev_delta)
        self.prev_delta = delta
        self.k += 1

        terminated = abs(self.x[0]) > 2.0
        truncated = self.k >= self.max_steps

        info = {}
        return self.x.astype(np.float32), reward, terminated, truncated, info
