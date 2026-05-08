from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    m: float
    Iz: float
    lf: float
    lr: float
    Cf: float
    Cr: float
    vx_nominal: float


class BicycleModel:
    """
    Small-angle lateral bicycle model around constant longitudinal speed.
    State x = [e_y, e_psi, v_y, r]
    You can later extend it to [e_y, e_psi, v, r, beta].
    """

    def __init__(self, params: VehicleParams, dt: float):
        self.p = params
        self.dt = dt

    def continuous_dynamics(self, x: np.ndarray, delta: float, vx: float | None = None) -> np.ndarray:
        vx = self.p.vx_nominal if vx is None else max(vx, 0.1)
        ey, epsi, vy, r = x

        m, Iz = self.p.m, self.p.Iz
        lf, lr = self.p.lf, self.p.lr
        Cf, Cr = self.p.Cf, self.p.Cr

        a11 = -(2 * Cf + 2 * Cr) / (m * vx)
        a12 = -(vx + (2 * Cf * lf - 2 * Cr * lr) / (m * vx))
        a21 = -(2 * Cf * lf - 2 * Cr * lr) / (Iz * vx)
        a22 = -(2 * Cf * lf**2 + 2 * Cr * lr**2) / (Iz * vx)
        b1 = 2 * Cf / m
        b2 = 2 * Cf * lf / Iz

        ey_dot = vy + vx * epsi
        epsi_dot = r
        vy_dot = a11 * vy + a12 * r + b1 * delta
        r_dot = a21 * vy + a22 * r + b2 * delta

        return np.array([ey_dot, epsi_dot, vy_dot, r_dot], dtype=float)

    def step(self, x: np.ndarray, delta: float, vx: float | None = None) -> np.ndarray:
        xdot = self.continuous_dynamics(x, delta, vx)
        return x + self.dt * xdot
