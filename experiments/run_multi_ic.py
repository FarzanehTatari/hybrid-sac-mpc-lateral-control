"""
Multi-initial-condition evaluation across the 5 retrained SAC seeds.

Loops over the 5 SAC checkpoints produced by run_multi_seed.py
(results/models/sac_seed{0..4}.zip) and evaluates the four controllers
on a 5x5 grid of initial conditions in (e_y, e_psi). Aggregates over
both seeds and ICs.

Run from project root:
    python -m experiments.run_multi_ic
Outputs:
    results/tables/multi_ic_raw.csv      (one row per seed per IC per ctrl)
    results/tables/multi_ic_summary.csv  (mean and std per controller)
"""
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from stable_baselines3 import SAC

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))


def rollout(controller, model, x0, n_steps=400, sac_model=None, use_sac=False,
            mpc_for_hybrid=None):
    x = x0.copy().astype(float)
    controller.reset()
    if mpc_for_hybrid is not None:
        mpc_for_hybrid.reset()
    log = {"ey": [], "delta": []}
    for k in range(n_steps):
        if use_sac:
            d_sac, _ = sac_model.predict(x.astype(np.float32), deterministic=True)
            kw = {"delta_sac": float(d_sac[0])}
            if mpc_for_hybrid is not None:
                kw["delta_mpc"] = float(mpc_for_hybrid.control(x, k))
            delta = controller.control(x, k, **kw)
        else:
            delta = controller.control(x, k)
        x = model.step(x, delta)
        log["ey"].append(x[0])
        log["delta"].append(delta)
    return log


def rollout_sac(sac_model, model, x0, n_steps=400):
    x = x0.copy().astype(float)
    log = {"ey": [], "delta": []}
    for k in range(n_steps):
        action, _ = sac_model.predict(x.astype(np.float32), deterministic=True)
        delta = float(action[0])
        x = model.step(x, delta)
        log["ey"].append(x[0])
        log["delta"].append(delta)
    return log


def build_plant_and_modular():
    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0, Cr=80000.0, vx_nominal=15.0,
    )
    model = BicycleModel(params, dt=0.05)

    pid = PIDBaseline()

    def _make_mpc():
        m = MPCController(horizon=15)
        m.set_weights(
            Q=np.diag([14, 5, 1, 2]),
            Qf=np.diag([14, 5, 1, 2]),
            R=np.array([[0.06]]),
            Rd=np.array([[1.5]]),
        )
        return m

    mpc = _make_mpc()
    # Separate MPC instance used internally by the Hybrid blend so its
    # warm-start state is independent of the standalone MPC rollout.
    mpc_hybrid = _make_mpc()

    hybrid = MPCFilterController(lambda_sac=12.0)
    return model, pid, mpc, mpc_hybrid, hybrid


def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    ey_grid    = [-0.4, -0.2, 0.0, 0.2, 0.4]
    epsi_grid  = [-0.10, -0.05, 0.0, 0.05, 0.10]
    ic_list    = list(product(ey_grid, epsi_grid))   # 25 ICs

    rows = []
    for seed in SEEDS:
        ckpt = f"results/models/sac_seed{seed}"
        print(f"[seed {seed}] loading {ckpt}.zip ...")
        sac = SAC.load(ckpt)

        # Modular controllers don't depend on the SAC seed, but to keep
        # the per-seed CSV self-contained we still log them per seed.
        model, pid, mpc, mpc_hybrid, hybrid = build_plant_and_modular()

        for (ey0, ep0) in ic_list:
            x0 = np.array([ey0, ep0, 0.0, 0.0])
            runs = {
                "PID":    rollout(pid, model, x0),
                "MPC":    rollout(mpc, model, x0),
                "SAC":    rollout_sac(sac, model, x0),
                "Hybrid": rollout(hybrid, model, x0,
                                  sac_model=sac, use_sac=True,
                                  mpc_for_hybrid=mpc_hybrid),
            }
            for name, log in runs.items():
                s = summarize_run(log)
                rows.append({
                    "seed": seed,
                    "ey0": ey0,
                    "epsi0": ep0,
                    "controller": name,
                    "rmse_ey": s["rmse_ey"],
                    "max_abs_ey": s["max_abs_ey"],
                    "mean_abs_delta": s["mean_abs_delta"],
                    "final_ey": log["ey"][-1],
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/multi_ic_raw.csv", index=False)

    # Aggregate across (seed, IC) for each controller.
    summary = (
        df.groupby("controller")
          .agg(rmse_ey_mean=("rmse_ey", "mean"),
               rmse_ey_std=("rmse_ey", "std"),
               max_abs_ey_mean=("max_abs_ey", "mean"),
               max_abs_ey_std=("max_abs_ey", "std"),
               mean_abs_delta_mean=("mean_abs_delta", "mean"),
               mean_abs_delta_std=("mean_abs_delta", "std"))
          .reset_index()
    )
    summary.to_csv("results/tables/multi_ic_summary.csv", index=False)

    print(f"\nMulti-IC summary "
          f"(mean +/- std over {N_SEEDS} seeds x {len(ic_list)} ICs):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
