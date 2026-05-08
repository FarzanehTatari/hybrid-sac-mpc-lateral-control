"""
Multi-seed SAC training and aggregated evaluation.

Trains SAC under N_SEEDS different seeds (default 5) and evaluates
PID, MPC, SAC, and Hybrid on the nominal initial-error recovery task
for each seed. Writes per-seed and aggregated CSVs.

Run from project root:
    python -m experiments.run_multi_seed
Outputs:
    results/models/sac_seed{S}.zip   (one per seed)
    results/tables/multi_seed_raw.csv
    results/tables/multi_seed_summary.csv

Compute budget: 150k SB3 SAC steps per seed with randomized initial
conditions in rl/env.py (ey ~ U[-0.4, 0.4], epsi ~ U[-0.10, 0.10]).
On a CPU laptop this is typically 15-30 minutes per seed;
total ~1.5-2.5 hours for 5 seeds.
"""
import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from stable_baselines3 import SAC

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run
from rl.env import VehicleLateralEnv

N_SEEDS = 5
TOTAL_TIMESTEPS = 150000   # bumped from 30k to reduce seed variance
SEEDS = list(range(N_SEEDS))


def set_global_seed(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def train_one(seed):
    set_global_seed(seed)
    env = VehicleLateralEnv()
    env.action_space.seed(seed)
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            pass

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_starts=1000,
        buffer_size=50000,
        seed=seed,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    out = f"results/models/sac_seed{seed}"
    model.save(out)
    return out


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


def evaluate_seed(seed):
    sac = SAC.load(f"results/models/sac_seed{seed}")
    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0, Cr=80000.0, vx_nominal=15.0,
    )
    model = BicycleModel(params, dt=0.05)
    x0 = np.array([0.2, 0.05, 0.0, 0.0])

    pid = PIDBaseline()
    mpc = MPCController(horizon=15)
    mpc.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]]),
    )
    # Separate MPC instance used internally by the Hybrid blend so its
    # warm-start state is independent of the standalone MPC rollout.
    mpc_hybrid = MPCController(horizon=15)
    mpc_hybrid.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]]),
    )
    hybrid = MPCFilterController(lambda_sac=12.0)

    runs = {
        "PID":    rollout(pid, model, x0),
        "MPC":    rollout(mpc, model, x0),
        "SAC":    rollout_sac(sac, model, x0),
        "Hybrid": rollout(hybrid, model, x0, sac_model=sac, use_sac=True,
                          mpc_for_hybrid=mpc_hybrid),
    }

    rows = []
    for name, log in runs.items():
        s = summarize_run(log)
        rows.append({
            "seed": seed,
            "controller": name,
            "rmse_ey": s["rmse_ey"],
            "max_abs_ey": s["max_abs_ey"],
            "mean_abs_delta": s["mean_abs_delta"],
            "final_ey": log["ey"][-1],
        })
    return rows


def main():
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    print(f"Training {N_SEEDS} SAC seeds, {TOTAL_TIMESTEPS} steps each...")
    for s in SEEDS:
        print(f"  seed {s} ...")
        train_one(s)

    print("\nEvaluating all four controllers per seed...")
    all_rows = []
    for s in SEEDS:
        all_rows.extend(evaluate_seed(s))

    df = pd.DataFrame(all_rows)
    df.to_csv("results/tables/multi_seed_raw.csv", index=False)

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
    summary.to_csv("results/tables/multi_seed_summary.csv", index=False)

    print("\nMulti-seed summary (mean +/- std over seeds):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
