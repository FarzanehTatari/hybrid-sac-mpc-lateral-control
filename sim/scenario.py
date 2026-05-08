from dataclasses import dataclass
import numpy as np


@dataclass
class Scenario:
    name: str = "nominal"
    process_noise_std: float = 0.0
    cf_scale: float = 1.0
    cr_scale: float = 1.0


def apply_process_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return x
    return x + rng.normal(0.0, std, size=x.shape)
