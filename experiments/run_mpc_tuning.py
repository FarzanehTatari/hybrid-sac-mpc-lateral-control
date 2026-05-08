import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from experiments.metrics import summarize_run


def rollout(controller, model, x0, n_steps=400):
    x = x0.copy()
    controller.reset()

    log = {"ey": [], "delta": []} # we log only these two, because we only need them in the score equation

    for k in range(n_steps):
        delta = controller.control(x, k)
        x = model.step(x, delta)

        log["ey"].append(x[0])
        log["delta"].append(delta)

    return log


def main():
    Path("results").mkdir(exist_ok=True)

    # Nominal plant
    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0, Cr=80000.0, vx_nominal=15.0
    )
    model = BicycleModel(params, dt=0.05)

    x0 = np.array([0.2, 0.05, 0.0, 0.0])

    # Local sweep (around your option 2)
    q_ey_vals = [10, 12, 14]
    q_epsi_vals = [5, 6, 7]
    q_r_vals = [2, 3, 4]
    R_vals = [0.06, 0.08, 0.10]
    Rd_vals = [1.5, 2, 3]

    results = []

    for q_ey, q_epsi, q_r, Rv, Rdv in product(
        q_ey_vals, q_epsi_vals, q_r_vals, R_vals, Rd_vals
    ):
        mpc = MPCController()

        Q = np.diag([q_ey, q_epsi, 1.0, q_r])
        Qf = Q.copy()
        R = np.array([[Rv]])
        Rd = np.array([[Rdv]])

        mpc.set_weights(Q=Q, Qf=Qf, R=R, Rd=Rd)

        log = rollout(mpc, model, x0)
        s = summarize_run(log)

        score = s["rmse_ey"] + 0.5 * s["max_abs_ey"] + 0.2 * s["mean_abs_delta"]

        results.append({
            "q_ey": q_ey,
            "q_epsi": q_epsi,
            "q_r": q_r,
            "R": Rv,
            "Rd": Rdv,
            "rmse": s["rmse_ey"],
            "max": s["max_abs_ey"],
            "mean_delta": s["mean_abs_delta"],
            "score": score
        })

    df = pd.DataFrame(results).sort_values("score")
    df.to_csv("results/mpc_tuning.csv", index=False)

    print("\nTop 10:\n")
    print(df.head(10))


if __name__ == "__main__":
    main()