import numpy as np
from sim.vehicle_model import BicycleModel, VehicleParams


def test_step_shape():
    params = VehicleParams(1600.0, 2500.0, 1.2, 1.6, 80000.0, 80000.0, 15.0)
    model = BicycleModel(params, dt=0.05)
    x = np.zeros(4)
    x_next = model.step(x, 0.01)
    assert x_next.shape == (4,)
