"""
Render the trajectory-comparison figure (Fig. 1 in the paper) using
the SAC checkpoint whose nominal-IC RMSE is closest to the 5-seed
median RMSE. This is the principled "representative seed" choice:
not cherry-picked, reproducible from the multi-seed CSV.

Run from project root:
    python -m experiments.run_final_results

Inputs:
    results/tables/multi_seed_raw.csv     (per-seed SAC RMSE)
    results/models/sac_seed{0..4}.zip     (the 5 retrained checkpoints)

Outputs:
    results/figures/final_controller_comparison.png
    results/tables/final_controller_metrics.csv

After running, copy the figure into the paper repo:
    cp results/figures/final_controller_comparison.png \\
       <paper-repo>/paper/figures/final_controller_comparison.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run


def pick_median_seed():
    """Return the seed (int) whose SAC nominal-IC RMSE is closest to the
    5-seed median. Reads from results/tables/multi_seed_raw.csv."""
    csv_path = Path("results/tables/multi_seed_raw.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run "
            "`python -m experiments.run_multi_seed` first."
        )
    df = pd.read_csv(csv_path)
    sac = df[df["controller"] == "SAC"].copy()
    if len(sac) == 0:
        raise RuntimeError("No SAC rows in multi_seed_raw.csv")
    median_rmse = sac["rmse_ey"].median()
    sac["abs_dev"] = (sac["rmse_ey"] - median_rmse).abs()
    chosen = sac.sort_values("abs_dev").iloc[0]
    print(
        f"[median-seed] SAC per-seed RMSE: "
        f"{sorted(sac['rmse_ey'].round(6).tolist())}"
    )
    print(
        f"[median-seed] median = {median_rmse:.6f} m, "
        f"chosen seed = {int(chosen['seed'])} "
        f"(RMSE = {chosen['rmse_ey']:.6f} m, "
        f"|dev from median| = {chosen['abs_dev']:.2e} m)"
    )
    return int(chosen["seed"])


def rollout_controller(controller, model, x0, n_steps=400,
                       sac_model=None, use_sac=False,
                       mpc_for_hybrid=None):
    x = x0.copy().astype(float)
    controller.reset()
    if mpc_for_hybrid is not None:
        mpc_for_hybrid.reset()
    log = {"t": [], "ey": [], "epsi": [], "delta": []}
    for k in range(n_steps):
        if use_sac:
            action, _ = sac_model.predict(x.astype(np.float32),
                                          deterministic=True)
            kw = {"delta_sac": float(action[0])}
            if mpc_for_hybrid is not None:
                kw["delta_mpc"] = float(mpc_for_hybrid.control(x, k))
            delta = controller.control(x, k, **kw)
        else:
            delta = controller.control(x, k)
        x = model.step(x, delta)
        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)
    return log


def rollout_sac(sac_model, model, x0, n_steps=400):
    x = x0.copy().astype(float)
    log = {"t": [], "ey": [], "epsi": [], "delta": []}
    for k in range(n_steps):
        action, _ = sac_model.predict(x.astype(np.float32),
                                      deterministic=True)
        delta = float(action[0])
        x = model.step(x, delta)
        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)
    return log


def plot_final(logs, chosen_seed):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for name, log in logs.items():
        axs[0].plot(log["t"], log["ey"], label=name)
    axs[0].set_ylabel(r"$e_y$ [m]")
    axs[0].set_title("Lateral Error")
    axs[0].grid(True)
    axs[0].legend()

    for name, log in logs.items():
        axs[1].plot(log["t"], log["epsi"], label=name)
    axs[1].set_ylabel(r"$e_\psi$ [rad]")
    axs[1].set_title("Heading Error")
    axs[1].grid(True)
    axs[1].legend()

    for name, log in logs.items():
        axs[2].plot(log["t"], log["delta"], label=name)
    axs[2].set_ylabel(r"$\delta$ [rad]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_title("Steering Input")
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle(
        f"Final Controller Comparison "
        f"(SAC seed {chosen_seed}, multi-seed median checkpoint)"
    )
    fig.tight_layout()
    return fig


def build_table(logs):
    rows = []
    for name, log in logs.items():
        s = summarize_run(log)
        rows.append({
            "Controller": name,
            "RMSE e_y": s["rmse_ey"],
            "Max |e_y|": s["max_abs_ey"],
            "Mean |delta|": s["mean_abs_delta"],
            "Final e_y": log["ey"][-1],
        })
    return pd.DataFrame(rows)


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    chosen_seed = pick_median_seed()
    sac_path = f"results/models/sac_seed{chosen_seed}"
    print(f"[load] SAC checkpoint: {sac_path}.zip")
    sac_model = SAC.load(sac_path)

    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0, Cr=80000.0, vx_nominal=15.0,
    )
    model = BicycleModel(params, dt=0.05)
    x0 = np.array([0.2, 0.05, 0.0, 0.0], dtype=float)
    n_steps = 400

    pid = PIDBaseline()
    mpc = MPCController(horizon=15)
    mpc.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]]),
    )
    # Separate MPC instance used internally by the Hybrid blend.
    mpc_hybrid = MPCController(horizon=15)
    mpc_hybrid.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]]),
    )
    hybrid = MPCFilterController(lambda_sac=12)

    logs = {
        "PID":    rollout_controller(pid, model, x0, n_steps),
        "MPC":    rollout_controller(mpc, model, x0, n_steps),
        "SAC":    rollout_sac(sac_model, model, x0, n_steps),
        "Hybrid": rollout_controller(hybrid, model, x0, n_steps,
                                     sac_model=sac_model, use_sac=True,
                                     mpc_for_hybrid=mpc_hybrid),
    }

    fig = plot_final(logs, chosen_seed)
    out_png = "results/figures/final_controller_comparison.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[save] {out_png}")

    table = build_table(logs)
    out_csv = "results/tables/final_controller_metrics.csv"
    table.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")

    print("\nFinal controller comparison table:\n")
    print(table.to_string(index=False))

    print(
        f"\n[next] Copy the figure into the paper repo:\n"
        f"    cp {out_png} <paper-repo>/paper/figures/"
        f"final_controller_comparison.png"
    )


if __name__ == "__main__":
    main()
