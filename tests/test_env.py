from rl.env import VehicleLateralEnv


def test_env_reset_step():
    env = VehicleLateralEnv()
    obs, info = env.reset()
    assert obs.shape == (4,)
    obs, reward, terminated, truncated, info = env.step([0.0])
    assert obs.shape == (4,)
