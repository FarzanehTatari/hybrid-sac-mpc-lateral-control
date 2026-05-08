from stable_baselines3 import SAC


def load_sac_policy(model_path: str):
    return SAC.load(model_path)
