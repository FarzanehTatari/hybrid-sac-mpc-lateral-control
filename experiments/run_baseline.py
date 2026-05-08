from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sim.vehicle_model import BicycleModel, VehicleParams
from controllers.pid_baseline import PIDBaseline
from utils.plotting import plot_run
from experiments.metrics import summarize_run


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    params = VehicleParams(
        m=1600.0, Iz=2500.0, lf=1.2, lr=1.6, Cf=80000.0, Cr=80000.0, vx_nominal=15.0
    )
    model = BicycleModel(params, dt=0.05)
    controller = PIDBaseline()

    x = np.array([0.2, 0.05, 0.0, 0.0], dtype=float)
    log = {"t": [], "ey": [], "epsi": [], "delta": []}

    for k in range(400):
        delta = controller.control(x, k)
        x = model.step(x, delta)

        log["t"].append(k * model.dt)
        log["ey"].append(x[0])
        log["epsi"].append(x[1])
        log["delta"].append(delta)

    print(summarize_run(log))
    fig = plot_run(log, title="PID baseline")
    fig.savefig("results/figures/pid_baseline.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
