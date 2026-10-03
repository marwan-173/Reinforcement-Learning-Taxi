import gymnasium as gym


def create_taxi_env(render_mode=None, seed=None): # Create and return the Taxi environment

    env = gym.make("Taxi-v4", is_rainy=False, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env