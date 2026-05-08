import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from controllers.linear_mpc_controller_w_Tnng_Rbst import MPCController
from controllers.mpc_filter_controller import MPCFilterController
from experiments.metrics import summarize_run
from stable_baselines3 import SAC


def rollout(controller, model, x0, n_steps=400, noise_std=None, rng=None, use_sac=False, sac_model=None):
    x = x0.copy()
    controller.reset()

    if rng is None:
        rng = np.random.default_rng(42)

    log = {"t": [], "ey": [], "epsi": [], "delta": []}

    for k in range(n_steps):
        if use_sac:
            obs = x.astype(np.float32)
            delta_sac, _ = sac_model.predict(obs, deterministic=True)
            delta_sac = float(delta_sac[0])
            delta = controller.control(x, k, delta_sac=delta_sac)
        else:
            delta = controller.control(x, k)

        x = model.step(x, delta)

        if noise_std is not None:
            noise = np.array([
                rng.normal(0, noise_std[0]),
                rng.normal(0, noise_std[1]),
                rng.normal(0, noise_std[2]),
                rng.normal(0, noise_std[3]),
            ])
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
        obs = x.astype(np.float32)
        action, _ = sac_model.predict(obs, deterministic=True)
        delta = float(action[0])

        x = model.step(x, delta)

        if noise_std is not None:
            noise = np.array([
                rng.normal(0, noise_std[0]),
                rng.normal(0, noise_std[1]),
                rng.normal(0, noise_std[2]),
                rng.normal(0, noise_std[3]),
            ])
            x = x + noise

        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)

    return log

def run_case(name, x0, noise_std=None, cf_scale=1.0):
    Path("results").mkdir(exist_ok=True)

    sac_model = SAC.load("results/models/sac_vehicle_lateral")

    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6,
        Cf=80000.0 * cf_scale,
        Cr=80000.0 * cf_scale,
        vx_nominal=15.0
    )
    model = BicycleModel(params, dt=0.05)

    # Controllers
    pid = PIDBaseline()

    mpc = MPCController()
    mpc.set_weights(
        Q=np.diag([14, 5, 1, 2]),
        Qf=np.diag([14, 5, 1, 2]),
        R=np.array([[0.06]]),
        Rd=np.array([[1.5]])
    )

    hybrid = MPCFilterController(lambda_sac=10.0)

    rng = np.random.default_rng(42)

    pid_log = rollout(pid, model, x0, noise_std=noise_std, rng=rng)
    mpc_log = rollout(mpc, model, x0, noise_std=noise_std, rng=rng)
    sac_log = rollout_sac(sac_model, model, x0, noise_std=noise_std, rng=rng)
    hybrid_log = rollout(hybrid, model, x0, noise_std=noise_std, rng=rng, use_sac=True, sac_model=sac_model)

    print(f"\n=== {name} ===")
    print("PID:", summarize_run(pid_log))
    print("MPC:", summarize_run(mpc_log))
    print("SAC:", summarize_run(sac_log))
    print("Hybrid:", summarize_run(hybrid_log))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pid_log["t"], pid_log["ey"], label="PID")
    plt.plot(mpc_log["t"], mpc_log["ey"], label="MPC")
    plt.plot(sac_log["t"], sac_log["ey"], label="SAC")
    plt.plot(hybrid_log["t"], hybrid_log["ey"], label="Hybrid")
    plt.title(name)
    plt.xlabel("time [s]")
    plt.ylabel("e_y [m]")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    # Case A
    run_case("Case A: large ey", np.array([0.4, 0.05, 0, 0]))

    # Case B
    run_case("Case B: large epsi", np.array([0.2, 0.1, 0, 0]))

    # Case C
    run_case(
        "Case C: noise",
        np.array([0.2, 0.05, 0, 0]),
        noise_std=[0.001, 0.0005, 0.005, 0.002]
    )

    # Case D
    run_case(
        "Case D: mismatch",
        np.array([0.2, 0.05, 0, 0]),
        cf_scale=0.8
    )


if __name__ == "__main__":
    main()