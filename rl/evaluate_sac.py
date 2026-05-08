from stable_baselines3 import SAC
from rl.env import VehicleLateralEnv


def main():
    env = VehicleLateralEnv()
    model = SAC.load("results/models/sac_vehicle_lateral")
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Evaluation total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
