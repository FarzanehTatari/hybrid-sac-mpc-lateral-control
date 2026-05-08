import numpy as np


def euler_step(x: np.ndarray, xdot: np.ndarray, dt: float) -> np.ndarray:
    return x + dt * xdot
