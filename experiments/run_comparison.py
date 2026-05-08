from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run
from stable_baselines3 import SAC


def rollout(controller, model, x0, n_steps=400, use_sac=False, sac_model=None):
    x = x0.copy().astype(float)
    controller.reset()

    log = {
        "t": [],
        "ey": [],
        "epsi": [],
        "delta": [],
    }

    for k in range(n_steps):
        if use_sac:
            # Placeholder SAC proposal for now.
            # Later replace with actual SAC policy output.
            obs = x.astype(np.float32)
            delta_sac, _ = sac_model.predict(obs, deterministic=True)
            delta_sac = float(delta_sac[0])
            delta = controller.control(x, k, delta_sac=delta_sac)
        else:
            delta = controller.control(x, k)

        x = model.step(x, delta)

        #noise = np.array([
        #np.random.normal(0.0, 0.001),   # e_y
        #np.random.normal(0.0, 0.0005),  # e_psi
        #np.random.normal(0.0, 0.005),   # v_y
        #np.random.normal(0.0, 0.002),   # r
        #])
        #x = x + noise


        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)

    return log

def rollout_sac(sac_model, model, x0, n_steps=400):
    x = x0.copy().astype(float)

    log = {"t": [], "ey": [], "epsi": [], "delta": []}

    for k in range(n_steps):
        obs = x.astype(np.float32)
        action, _ = sac_model.predict(obs, deterministic=True)
        delta = float(action[0])

        x = model.step(x, delta)

        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)

    return log

def plot_comparison(pid_log, mpc_log, sac_log, hybrid_log):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(pid_log["t"], pid_log["ey"], label="PID")
    axs[0].plot(mpc_log["t"], mpc_log["ey"], label="MPC")
    axs[0].plot(sac_log["t"], sac_log["ey"], label="SAC")
    axs[0].plot(hybrid_log["t"], hybrid_log["ey"], label="Hybrid")
    axs[0].set_ylabel("e_y [m]")
    axs[0].set_title("Lateral error comparison")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(pid_log["t"], pid_log["epsi"], label="PID")
    axs[1].plot(mpc_log["t"], mpc_log["epsi"], label="MPC")
    axs[1].plot(sac_log["t"], sac_log["epsi"], label="SAC")
    axs[1].plot(hybrid_log["t"], hybrid_log["epsi"], label="Hybrid")
    axs[1].set_ylabel("e_psi [rad]")
    axs[1].set_title("Heading error comparison")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(pid_log["t"], pid_log["delta"], label="PID")
    axs[2].plot(mpc_log["t"], mpc_log["delta"], label="MPC")
    axs[2].plot(sac_log["t"], sac_log["delta"], label="SAC")
    axs[2].plot(hybrid_log["t"], hybrid_log["delta"], label="Hybrid")
    axs[2].set_ylabel("delta [rad]")
    axs[2].set_xlabel("time [s]")
    axs[2].set_title("Steering input comparison")
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle("PID vs MPC vs Hybrid")
    fig.tight_layout()
    return fig


def print_summary(name, log):
    summary = summarize_run(log)
    print(f"\n{name} summary")
    print(f"  RMSE e_y      : {summary['rmse_ey']:.6f}")
    print(f"  Max |e_y|     : {summary['max_abs_ey']:.6f}")
    print(f"  Mean |delta|  : {summary['mean_abs_delta']:.6f}")
    print(f"  Final e_y     : {log['ey'][-1]:.6e}")


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    params = VehicleParams(
        m=1600.0,
        Iz=2500.0,
        lf=1.2,
        lr=1.6,
        Cf=80000.0 * 0.8,
        Cr=80000.0 * 0.8,
        vx_nominal=15.0,
    )

    model = BicycleModel(params, dt=0.05)
    
    x0 = np.array([0.2, 0.05, 0.0, 0.0], dtype=float)
    n_steps = 400
    
    sac_model = SAC.load("results/models/sac_vehicle_lateral")
    pid = PIDBaseline()
    mpc = MPCController(horizon=15)
    hybrid = MPCFilterController(lambda_sac=5.0)

    sac_log = rollout_sac(sac_model, model, x0, n_steps=n_steps)
    pid_log = rollout(pid, model, x0, n_steps=n_steps, use_sac=False)
    mpc_log = rollout(mpc, model, x0, n_steps=n_steps, use_sac=False)
    hybrid_log = rollout(hybrid, model, x0, n_steps=n_steps, use_sac=True, sac_model=sac_model)

    print_summary("PID", pid_log)
    print_summary("MPC", mpc_log)
    print_summary("SAC", sac_log)
    print_summary("Hybrid", hybrid_log)

    fig = plot_comparison(pid_log, mpc_log, sac_log, hybrid_log)
    fig.savefig("results/figures/controller_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()