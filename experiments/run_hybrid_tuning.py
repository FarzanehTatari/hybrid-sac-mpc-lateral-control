"""
Hybrid blending sweep over lambda on the nominal task, aggregated
over the 5 retrained SAC seeds (sac_seed0..4) and using the
MPC-blend formulation of mpc_filter_controller.py.

Run from project root:
    python -m experiments.run_hybrid_tuning
Outputs:
    results/tables/hybrid_lambda_tuning_raw.csv      (per seed, per lambda)
    results/tables/hybrid_lambda_tuning_summary.csv  (mean per lambda)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))


def make_mpc():
    mpc = MPCController(horizon=15)
    mpc.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]]),
    )
    return mpc


def rollout_hybrid(lambda_sac, sac_model, model, x0, n_steps=400):
    controller = MPCFilterController(lambda_sac=lambda_sac)
    controller.reset()
    mpc_for_hybrid = make_mpc()
    mpc_for_hybrid.reset()

    x = x0.copy().astype(float)
    log = {"ey": [], "delta": []}

    for k in range(n_steps):
        obs = x.astype(np.float32)
        action, _ = sac_model.predict(obs, deterministic=True)
        delta_sac = float(action[0])
        delta_mpc = float(mpc_for_hybrid.control(x, k))

        delta = controller.control(
            x, k, delta_sac=delta_sac, delta_mpc=delta_mpc
        )
        x = model.step(x, delta)

        log["ey"].append(x[0])
        log["delta"].append(delta)

    return log


def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    params = VehicleParams(
        m=1600.0,
        Iz=2500.0,
        lf=1.2,
        lr=1.6,
        Cf=80000.0,
        Cr=80000.0,
        vx_nominal=15.0,
    )

    model = BicycleModel(params, dt=0.05)
    x0 = np.array([0.2, 0.05, 0.0, 0.0])
    lambda_values = [1, 2, 3, 5, 8, 10, 12, 15]

    rows = []

    for seed in SEEDS:
        ckpt = f"results/models/sac_seed{seed}"
        print(f"[seed {seed}] loading {ckpt}.zip ...")
        sac_model = SAC.load(ckpt)

        for lam in lambda_values:
            log = rollout_hybrid(lam, sac_model, model, x0)
            s = summarize_run(log)
            score = (
                s["rmse_ey"]
                + 0.5 * s["max_abs_ey"]
                + 0.5 * s["mean_abs_delta"]
            )
            rows.append({
                "seed": seed,
                "lambda_sac": lam,
                "rmse_ey": s["rmse_ey"],
                "max_abs_ey": s["max_abs_ey"],
                "mean_abs_delta": s["mean_abs_delta"],
                "score": score,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/hybrid_lambda_tuning_raw.csv", index=False)

    summary = (
        df.groupby("lambda_sac")
          .agg(rmse_ey_mean=("rmse_ey", "mean"),
               rmse_ey_std=("rmse_ey", "std"),
               max_abs_ey_mean=("max_abs_ey", "mean"),
               mean_abs_delta_mean=("mean_abs_delta", "mean"),
               score_mean=("score", "mean"))
          .reset_index()
          .sort_values("lambda_sac")
    )
    summary.to_csv("results/tables/hybrid_lambda_tuning_summary.csv",
                   index=False)

    print("\nHybrid lambda tuning summary "
          f"(mean over {N_SEEDS} seeds):\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()