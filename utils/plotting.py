import matplotlib.pyplot as plt
import numpy as np


def plot_run(log: dict, title: str = "Run"):
    t = np.asarray(log["t"])
    ey = np.asarray(log["ey"])
    epsi = np.asarray(log["epsi"])
    delta = np.asarray(log["delta"])

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t, ey)
    axs[0].set_ylabel("e_y [m]")
    axs[0].grid(True)

    axs[1].plot(t, epsi)
    axs[1].set_ylabel("e_psi [rad]")
    axs[1].grid(True)

    axs[2].plot(t, delta)
    axs[2].set_ylabel("delta [rad]")
    axs[2].set_xlabel("time [s]")
    axs[2].grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
