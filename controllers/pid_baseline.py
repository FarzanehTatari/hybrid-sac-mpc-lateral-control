from controllers.base_controller import BaseController


class PIDBaseline(BaseController):
    def __init__(self, kp_y: float = 0.3, kp_psi: float = 0.8):
        self.kp_y = kp_y
        self.kp_psi = kp_psi

    def reset(self) -> None:
        return None

    def control(self, x, k: int, **kwargs) -> float:
        ey, epsi, vy, r = x
        delta = -self.kp_y * ey - self.kp_psi * epsi
        return float(delta)
