from src.environment import create_taxi_env
from src.agent import Agent
from src.rewards import base_reward
import src.config as config


def main():

    env = create_taxi_env(seed=config.SEED)
    agent = Agent(env=env, reward_function=base_reward, config=config, use_planning=True,  use_kernel=True,)
    metrics = agent.train()
    final = metrics[-1]

    print(f"Episodes completed: {config.NUM_EPISODES}")
    print(f"Final reward: {final.cumulative_reward:.2f}")
    print(f"Final episode length: {final.episode_length}")
    print(f"Final success: {final.success}")

    env.close()


if __name__ == "__main__":
    main()