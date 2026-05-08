"""
Quantitative robustness experiments across the 5 retrained SAC seeds.

Same four cases as run_robustness.py, but loops over the SAC seed
checkpoints produced by run_multi_seed.py
(results/models/sac_seed{0..4}.zip) and aggregates per-case
mean +/- std across seeds.

Run from project root:
    python -m experiments.run_robustness_quant
Outputs:
    results/tables/robustness_quant_raw.csv     (one row per seed per case per ctrl)
    results/tables/robustness_quant_summary.csv (mean +/- std per case per ctrl)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))


def rollout(controller, model, x0, n_steps=400, noise_std=None, rng=None,
            use_sac=False, sac_model=None, mpc_for_hybrid=None):
    x = x0.copy().astype(float)
    controller.reset()
    if mpc_for_hybrid is not None:
        mpc_for_hybrid.reset()
    if rng is None:
        rng = np.random.default_rng(42)

    log = {"t": [], "ey": [], "epsi": [], "delta": []}
    for k in range(n_steps):
        if use_sac:
            obs = x.astype(np.float32)
            d_sac, _ = sac_model.predict(obs, deterministic=True)
            kw = {"delta_sac": float(d_sac[0])}
            if mpc_for_hybrid is not None:
                kw["delta_mpc"] = float(mpc_for_hybrid.control(x, k))
            delta = controller.control(x, k, **kw)
        else:
            delta = controller.control(x, k)
        x = model.step(x, delta)
        if noise_std is not None:
            noise = np.array([rng.normal(0, s) for s in noise_std])
            x = x + noise
        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)
    return log


def rollout_sac(sac_model, model, x0, n_steps=400, noise_std=None, rng=None):
    x = x0.copy().astype(float)
    if rng is None:
        rng = np.random.default_rng(42)
    log = {"t": [], "ey": [], "epsi": [], "delta": []}
    for k in range(n_steps):
        action, _ = sac_model.predict(x.astype(np.float32), deterministic=True)
        delta = float(action[0])
        x = model.step(x, delta)
        if noise_std is not None:
            noise = np.array([rng.normal(0, s) for s in noise_std])
            x = x + noise
        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)
    return log


def build_controllers(cf_scale=1.0):
    """
    Build the plant and the four controllers used in each robustness case.

    Note: `cf_scale` only changes the *true plant* tire stiffness used for
    rollout. The MPC's internal model is kept at the nominal Cf/Cr (the
    "controller-model mismatch" framing). The Hybrid's internal MPC also
    uses the nominal model for the same reason — both MPC instances share
    the nominal controller model and only the plant differs.
    """
    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0 * cf_scale, Cr=80000.0 * cf_scale, vx_nominal=15.0,
    )
    model = BicycleModel(params, dt=0.05)

    pid = PIDBaseline()

    def _make_mpc():
        m = MPCController(horizon=15)  # nominal Cf, Cr by construction
        m.set_weights(
            Q=np.diag([14, 5, 1, 2]),
            Qf=np.diag([14, 5, 1, 2]),
            R=np.array([[0.06]]),
            Rd=np.array([[1.5]]),
        )
        return m

    mpc = _make_mpc()
    # Separate MPC instance used internally by the Hybrid blend.
    mpc_hybrid = _make_mpc()

    hybrid = MPCFilterController(lambda_sac=12.0)
    return model, pid, mpc, mpc_hybrid, hybrid


def run_case_for_seed(seed, sac_model, case_name, x0,
                      noise_std=None, cf_scale=1.0):
    model, pid, mpc, mpc_hybrid, hybrid = build_controllers(cf_scale=cf_scale)
    rng_seed = 42

    rows = []
    for ctrl_name, runner in [
        ("PID",    lambda: rollout(pid, model, x0, noise_std=noise_std,
                                   rng=np.random.default_rng(rng_seed))),
        ("MPC",    lambda: rollout(mpc, model, x0, noise_std=noise_std,
                                   rng=np.random.default_rng(rng_seed))),
        ("SAC",    lambda: rollout_sac(sac_model, model, x0,
                                       noise_std=noise_std,
                                       rng=np.random.default_rng(rng_seed))),
        ("Hybrid", lambda: rollout(hybrid, model, x0, noise_std=noise_std,
                                   rng=np.random.default_rng(rng_seed),
                                   use_sac=True, sac_model=sac_model,
                                   mpc_for_hybrid=mpc_hybrid)),
    ]:
        log = runner()
        s = summarize_run(log)
        rows.append({
            "seed": seed,
            "case": case_name,
            "controller": ctrl_name,
            "rmse_ey": s["rmse_ey"],
            "max_abs_ey": s["max_abs_ey"],
            "mean_abs_delta": s["mean_abs_delta"],
            "final_ey": log["ey"][-1],
        })
    return rows


def main():
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    cases = [
        ("A_large_ey",     dict(x0=np.array([0.4, 0.05, 0, 0]))),
        ("B_large_epsi",   dict(x0=np.array([0.2, 0.10, 0, 0]))),
        ("C_proc_noise",   dict(x0=np.array([0.2, 0.05, 0, 0]),
                                noise_std=[0.001, 0.0005, 0.005, 0.002])),
        ("D_cf_mismatch",  dict(x0=np.array([0.2, 0.05, 0, 0]), cf_scale=0.8)),
    ]

    all_rows = []
    for seed in SEEDS:
        ckpt = f"results/models/sac_seed{seed}"
        print(f"[seed {seed}] loading {ckpt}.zip ...")
        sac = SAC.load(ckpt)
        for case_name, kwargs in cases:
            all_rows.extend(
                run_case_for_seed(seed, sac, case_name, **kwargs)
            )

    df = pd.DataFrame(all_rows)
    df.to_csv("results/tables/robustness_quant_raw.csv", index=False)

    summary = (
        df.groupby(["case", "controller"])
          .agg(rmse_ey_mean=("rmse_ey", "mean"),
               rmse_ey_std=("rmse_ey", "std"),
               max_abs_ey_mean=("max_abs_ey", "mean"),
               max_abs_ey_std=("max_abs_ey", "std"),
               mean_abs_delta_mean=("mean_abs_delta", "mean"),
               mean_abs_delta_std=("mean_abs_delta", "std"))
          .reset_index()
    )
    summary.to_csv("results/tables/robustness_quant_summary.csv", index=False)

    print(f"\nRobustness summary "
          f"(mean +/- std over {N_SEEDS} seeds, per case per controller):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
