import numpy as np


def rmse(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def summarize_run(log: dict) -> dict:
    ey = np.asarray(log["ey"])
    delta = np.asarray(log["delta"])
    return {
        "rmse_ey": rmse(ey),
        "max_abs_ey": float(np.max(np.abs(ey))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }
