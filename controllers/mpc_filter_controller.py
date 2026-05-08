class MPCFilterController:
    """
    Hybrid controller (closed-form MPC-blend filter).

    Blends SAC's proposed action with the MPC's first-step optimum:
        alpha = lambda / (1 + lambda)
        delta_exec = (1 - alpha) * delta_mpc + alpha * delta_sac

    Inputs (passed via kwargs in control()):
        delta_sac : float
            SAC's proposed steering input at the current state.
        delta_mpc : float
            MPC's first-step optimum at the current state (the action
            MPC would apply if run alone).

    If `delta_mpc` is not provided, the blend falls back to a small
    feedback-gain tracking law on the state, which keeps the controller
    runnable without an MPC instance but is NOT the configuration used
    in the paper. The paper's hybrid always passes `delta_mpc`.
    """

    def __init__(self, lambda_sac: float = 5.0):
        self.lambda_sac = lambda_sac
        self.prev_delta = 0.0

    def reset(self) -> None:
        self.prev_delta = 0.0

    def control(self, x, k: int, **kwargs) -> float:
        delta_sac = kwargs.get("delta_sac", 0.0)

        if "delta_mpc" in kwargs:
            delta_tracking = float(kwargs["delta_mpc"])
        else:
            # Fallback feedback-gain tracking law (NOT used in the paper).
            ey, epsi, vy, r = x
            delta_tracking = -0.2 * ey - 0.7 * epsi - 0.05 * r

        alpha = self.lambda_sac / (1.0 + self.lambda_sac)
        delta_exec = (1.0 - alpha) * delta_tracking + alpha * delta_sac

        # optional safety clipping
        delta_exec = max(min(delta_exec, 0.4), -0.4)

        self.prev_delta = float(delta_exec)
        return float(delta_exec)
