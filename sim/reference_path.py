import numpy as np


def lane_change_reference(n_steps: int, dt: float, vx: float, amplitude: float = 1.0, length: float = 60.0) -> np.ndarray:
    """Returns desired lateral position profile y_ref over time."""
    t = np.arange(n_steps) * dt
    x = vx * t
    y_ref = amplitude * 0.5 * (1.0 + np.tanh((x - length / 2.0) / 5.0))
    return y_ref


def heading_from_lateral_profile(y_ref: np.ndarray, dt: float, vx: float) -> np.ndarray:
    dy_dt = np.gradient(y_ref, dt)
    return np.arctan2(dy_dt, vx)
