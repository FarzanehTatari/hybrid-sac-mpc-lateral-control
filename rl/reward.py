def compute_reward(x, delta: float, delta_prev: float) -> float:
    ey, epsi, vy, r = x
    reward = (
        -2.0 * ey**2
        -1.0 * epsi**2
        -0.05 * delta**2
        -0.1 * (delta - delta_prev) ** 2
    )
    return float(reward)
