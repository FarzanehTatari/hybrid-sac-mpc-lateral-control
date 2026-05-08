class MPCController:
    """
    Stub for baseline MPC.
    Replace the control() method with your OSQP or CasADi-based solver.
    """

    def __init__(self, horizon: int = 15):
        self.horizon = horizon
        self.prev_delta = 0.0

    def reset(self) -> None:
        self.prev_delta = 0.0

    def control(self, x, k: int, **kwargs) -> float:
        # TODO: build and solve QP/NLP and return first steering input.
        delta = -0.2 * x[0] - 0.7 * x[1] - 0.05 * x[3]
        self.prev_delta = float(delta)
        return float(delta)
