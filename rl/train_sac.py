from pathlib import Path
from stable_baselines3 import SAC
from rl.env import VehicleLateralEnv


def main():
    Path("results/models").mkdir(parents=True, exist_ok=True)
    env = VehicleLateralEnv()
    model = SAC("MlpPolicy", env, verbose=1, learning_starts=1000, buffer_size=50000)
    model.learn(total_timesteps=30000)
    model.save("results/models/sac_vehicle_lateral")


if __name__ == "__main__":
    main()
